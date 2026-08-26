import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. PARÂMETROS DO DRONE E LIMITES FÍSICOS
# ==========================================
Ixx = 44.56e-6
Iyy = 42.98e-6
Izz = 78.79e-6
Ir = 5.196e-08
m = 0.0469  # 46.9g
g = 9.80665

# ==========================================
# TEMPO DE PROCESSAMENTO DO LOOP DE CONTROLE (síncrono, mono-thread)
# Sem task assíncrona, tudo roda sequencialmente no mesmo laço:
#   sensores → Madgwick → aplica controle aos motores → calcula ganho SDRE
# O período do loop é a SOMA dos tempos abaixo (não um clock fixo assumido).
# Como o SDRE deste ciclo só termina DEPOIS da atuação dos motores, o ganho K
# usado em "aplica controle" é sempre o calculado no ciclo ANTERIOR.
# ==========================================
T_SENSOR_READ_S = 0.001  # leitura MPU6050 + QMC5883L via I2C
T_MADGWICK_CALC_S = 0.001  # filtro Madgwick MARG
T_SDRE_CALC_S = 0.032  # solve_discrete_are 6x6 — gargalo do ciclo
T_MOTOR_APPLY_S = 0.0005  # mixer + escrita do comando nos ESCs
DT_LOOP = (
    T_SENSOR_READ_S + T_MADGWICK_CALC_S + T_SDRE_CALC_S + T_MOTOR_APPLY_S
)  # ~12.5ms (~80Hz)

dt_pid = 0.05  # PID de posição — laço externo, mais lento que o de atitude
SENSOR_DELAY_S = 0.0015  # Atraso de leitura do sensor em segundos (configurável)

# Atraso intrínseco de acionamento dos motores: resposta eletromecânica do
# conjunto ESC+motor+hélice a uma mudança de comando (física do atuador, não
# tempo de CPU). Modelado como lag de 1ª ordem — mesma convenção de
# tau_motor_ms usada na identificação de parâmetros (identify_params.py).
MOTOR_LAG_TAU_S = 0.080

# Parâmetros Físicos dos Motores (Do main.cpp)
b_coeff = 2.98e-8  # Coeficiente de Empuxo
d_coeff = 0.05 * b_coeff  # Coeficiente de Arrasto
L_arm = 0.060  # 60mm
L_eff = L_arm * np.sin(np.pi / 4)  # Braço efetivo para config em 'X'

# LIMITES DE RPM (Saturação)
MAX_RPM = 26423.0
MAX_OMEGA = MAX_RPM * (2.0 * np.pi) / 60.0
MAX_OMEGA_SQ = MAX_OMEGA**2

# Matriz de Mistura (Mixer Matrix) para Quadricóptero em 'X'
M_mixer = np.array(
    [
        [b_coeff, b_coeff, b_coeff, b_coeff],  # Empuxo Total (Z)
        [
            -b_coeff * L_eff,
            -b_coeff * L_eff,
            b_coeff * L_eff,
            b_coeff * L_eff,
        ],  # Roll (X)
        [
            -b_coeff * L_eff,
            b_coeff * L_eff,
            b_coeff * L_eff,
            -b_coeff * L_eff,
        ],  # Pitch (Y)
        [-d_coeff, d_coeff, -d_coeff, d_coeff],  # Yaw (Z)
    ]
)
M_inv = np.linalg.inv(M_mixer)

# Matriz de Mistura REAIS (simulando desbalanceamento de até 5% nos motores)
error_percent = 0.03
np.random.seed(2)  # Para resultados reprodutíveis
b_coeffs_real = b_coeff * np.random.uniform(1 - error_percent, 1 + error_percent, 4)
d_coeffs_real = d_coeff * np.random.uniform(1 - error_percent, 1 + error_percent, 4)

M_mixer_real = np.array(
    [
        [b_coeffs_real[0], b_coeffs_real[1], b_coeffs_real[2], b_coeffs_real[3]],
        [
            -b_coeffs_real[0] * L_eff,
            -b_coeffs_real[1] * L_eff,
            b_coeffs_real[2] * L_eff,
            b_coeffs_real[3] * L_eff,
        ],
        [
            -b_coeffs_real[0] * L_eff,
            b_coeffs_real[1] * L_eff,
            b_coeffs_real[2] * L_eff,
            -b_coeffs_real[3] * L_eff,
        ],
        [-d_coeffs_real[0], d_coeffs_real[1], -d_coeffs_real[2], d_coeffs_real[3]],
    ]
)

# Arrasto translacional (força de amortecimento linear F = -k_d·v)
# Pequenos drones têm tipicamente 0.005–0.02 N·s/m.
# O arrasto é adicionado tanto na dinâmica quanto no modelo do acelerômetro,
# pois cria componente lateral no corpo que o Madgwick usa como cue de inclinação.
K_DRAG_TRANS = 0.01  # N·s/m

# Matrizes de Peso do SDRE (Q e R) — IDÊNTICOS ao main.cpp linhas 143-148
roll_max_rad = 45.0 * np.pi / 180.0  # main.cpp: roll_max_rad = 45° DEG_TO_RAD
pitch_max_rad = 45.0 * np.pi / 180.0  # main.cpp: pitch_max_rad = 45° DEG_TO_RAD
yaw_max_rad = 90.0 * np.pi / 180.0  # main.cpp: yaw_max_rad = 90° DEG_TO_RAD
p_max = 100.0 * np.pi / 180.0  # main.cpp: p_max = 100°/s
q_max = 100.0 * np.pi / 180.0  # main.cpp: q_max = 100°/s
r_max = 200.0 * np.pi / 180.0  # main.cpp: r_max = 200°/s

Q = np.diag(
    [
        1.0 / (roll_max_rad**2),
        1.0 / (pitch_max_rad**2),
        1.0 / (yaw_max_rad**2),
        1.0 / (p_max**2),
        1.0 / (q_max**2),
        1.0 / (r_max**2),
    ]
)
R_mat = np.eye(3)

max_tau_roll = 2 * b_coeff * L_eff * MAX_OMEGA * MAX_OMEGA
max_tau_pitch = 2 * b_coeff * L_eff * MAX_OMEGA * MAX_OMEGA
max_tau_yaw = 2 * d_coeff * MAX_OMEGA * MAX_OMEGA

R_mat = np.diag(
    [
        1.0 / (max_tau_roll**2),
        1.0 / (max_tau_pitch**2),
        1.0 / (max_tau_yaw**2),
    ]
)


# ==========================================
# 1B. SENSORES (MPU6050 + QMC5883L) E MADGWICK
# ==========================================
# Ruídos típicos do MPU6050 com DLPF @ ~98Hz
GYRO_NOISE_STD = 0.01  # rad/s (~0.57 °/s)
ACCEL_NOISE_STD = 0.10  # m/s²
GYRO_BIAS = np.array([0.003, -0.002, 0.004])  # rad/s
ACCEL_BIAS = np.array([0.05, -0.03, 0.08])  # m/s²

# Ruído do QMC5883L (saída normalizada)
MAG_NOISE_STD = 0.015

# Inclinação magnética típica (Brasil ≈ -30°). ENU/Z-up: bz = -sin(inc)
MAG_INCLINATION = np.deg2rad(-30.0)
MAG_WORLD = np.array([np.cos(MAG_INCLINATION), 0.0, -np.sin(MAG_INCLINATION)])
MAG_WORLD /= np.linalg.norm(MAG_WORLD)

# Ganho β do filtro Madgwick (compromisso giroscópio × accel/mag)
MADGWICK_BETA = 0.03


def madgwick_marg_update(q, gx, gy, gz, ax, ay, az, mx, my, mz, beta, dt):
    """Filtro Madgwick MARG (Magnetic, Angular Rate, Gravity).
    Convenção Z-up coerente com a simulação. q = [w, x, y, z].
    """
    q1, q2, q3, q4 = q  # q1=w, q2=x, q3=y, q4=z

    norm_a = np.sqrt(ax * ax + ay * ay + az * az)
    if norm_a < 1e-9:
        return q
    ax, ay, az = ax / norm_a, ay / norm_a, az / norm_a

    norm_m = np.sqrt(mx * mx + my * my + mz * mz)
    if norm_m < 1e-9:
        return q
    mx, my, mz = mx / norm_m, my / norm_m, mz / norm_m

    # Direção de referência do campo magnético terrestre (descontando inclinação atual)
    _2q1mx = 2 * q1 * mx
    _2q1my = 2 * q1 * my
    _2q1mz = 2 * q1 * mz
    _2q2mx = 2 * q2 * mx
    hx = (
        mx * q1 * q1
        - _2q1my * q4
        + _2q1mz * q3
        + mx * q2 * q2
        + 2 * q2 * my * q3
        + 2 * q2 * mz * q4
        - mx * q3 * q3
        - mx * q4 * q4
    )
    hy = (
        _2q1mx * q4
        + my * q1 * q1
        - _2q1mz * q2
        + _2q2mx * q3
        - my * q2 * q2
        + my * q3 * q3
        + 2 * q3 * mz * q4
        - my * q4 * q4
    )
    _2bx = np.sqrt(hx * hx + hy * hy)
    _2bz = (
        -_2q1mx * q3
        + _2q1my * q2
        + mz * q1 * q1
        + _2q2mx * q4
        - mz * q2 * q2
        + 2 * q3 * my * q4
        - mz * q3 * q3
        + mz * q4 * q4
    )
    _4bx = 2 * _2bx
    _4bz = 2 * _2bz

    # Gradiente da função objetivo (gravidade + bússola)
    s1 = (
        -2 * q3 * (2 * q2 * q4 - 2 * q1 * q3 - ax)
        + 2 * q2 * (2 * q1 * q2 + 2 * q3 * q4 - ay)
        - _2bz
        * q3
        * (_2bx * (0.5 - q3 * q3 - q4 * q4) + _2bz * (q2 * q4 - q1 * q3) - mx)
        + (-_2bx * q4 + _2bz * q2)
        * (_2bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - my)
        + _2bx
        * q3
        * (_2bx * (q1 * q3 + q2 * q4) + _2bz * (0.5 - q2 * q2 - q3 * q3) - mz)
    )
    s2 = (
        2 * q4 * (2 * q2 * q4 - 2 * q1 * q3 - ax)
        + 2 * q1 * (2 * q1 * q2 + 2 * q3 * q4 - ay)
        - 4 * q2 * (1 - 2 * q2 * q2 - 2 * q3 * q3 - az)
        + _2bz
        * q4
        * (_2bx * (0.5 - q3 * q3 - q4 * q4) + _2bz * (q2 * q4 - q1 * q3) - mx)
        + (_2bx * q3 + _2bz * q1)
        * (_2bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - my)
        + (_2bx * q4 - _4bz * q2)
        * (_2bx * (q1 * q3 + q2 * q4) + _2bz * (0.5 - q2 * q2 - q3 * q3) - mz)
    )
    s3 = (
        -2 * q1 * (2 * q2 * q4 - 2 * q1 * q3 - ax)
        + 2 * q4 * (2 * q1 * q2 + 2 * q3 * q4 - ay)
        - 4 * q3 * (1 - 2 * q2 * q2 - 2 * q3 * q3 - az)
        + (-_4bx * q3 - _2bz * q1)
        * (_2bx * (0.5 - q3 * q3 - q4 * q4) + _2bz * (q2 * q4 - q1 * q3) - mx)
        + (_2bx * q2 + _2bz * q4)
        * (_2bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - my)
        + (_2bx * q1 - _4bz * q3)
        * (_2bx * (q1 * q3 + q2 * q4) + _2bz * (0.5 - q2 * q2 - q3 * q3) - mz)
    )
    s4 = (
        2 * q2 * (2 * q2 * q4 - 2 * q1 * q3 - ax)
        + 2 * q3 * (2 * q1 * q2 + 2 * q3 * q4 - ay)
        + (-_4bx * q4 + _2bz * q2)
        * (_2bx * (0.5 - q3 * q3 - q4 * q4) + _2bz * (q2 * q4 - q1 * q3) - mx)
        + (-_2bx * q1 + _2bz * q3)
        * (_2bx * (q2 * q3 - q1 * q4) + _2bz * (q1 * q2 + q3 * q4) - my)
        + _2bx
        * q2
        * (_2bx * (q1 * q3 + q2 * q4) + _2bz * (0.5 - q2 * q2 - q3 * q3) - mz)
    )
    norm_s = np.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)
    if norm_s < 1e-12:
        return q
    s1 /= norm_s
    s2 /= norm_s
    s3 /= norm_s
    s4 /= norm_s

    # Taxa de variação do quaternion (giroscópio + correção pelo gradiente)
    qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - beta * s1
    qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - beta * s2
    qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - beta * s3
    qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - beta * s4

    q1 += qDot1 * dt
    q2 += qDot2 * dt
    q3 += qDot3 * dt
    q4 += qDot4 * dt
    n = np.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
    return np.array([q1, q2, q3, q4]) / n


def quat_to_euler(quat):
    """Quaternion [w,x,y,z] → (phi, theta, psi) na convenção ZYX."""
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    phi = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    theta = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    psi = np.arctan2(siny_cosp, cosy_cosp)
    return phi, theta, psi


def simulate_sensors(state, a_cg_world, R_wb):
    """Gera leituras simuladas do MPU6050 (acel + giro) e QMC5883L (mag).

    Acelerômetro mede força específica: f = R^T · (a_cg - g_world).
    Z-up: g_world = [0, 0, -g], logo f inclui o vetor de gravidade rotacionado no corpo.
    Quando nivelado (a_cg=0): accel_body = [0, 0, g].
    Quando inclinado (a_cg=0): accel_body = g·R^T·ẑ — cue de tilt para o Madgwick.
    """
    p_b, q_b, r_b = state[9], state[10], state[11]

    g_world = np.array([0.0, 0.0, -g])
    f_specific_world = a_cg_world - g_world
    accel_body = R_wb.T @ f_specific_world
    accel_meas = accel_body + ACCEL_BIAS + np.random.normal(0.0, ACCEL_NOISE_STD, 3)

    gyro_meas = (
        np.array([p_b, q_b, r_b]) + GYRO_BIAS + np.random.normal(0.0, GYRO_NOISE_STD, 3)
    )

    mag_body = R_wb.T @ MAG_WORLD
    mag_meas = mag_body + np.random.normal(0.0, MAG_NOISE_STD, 3)

    return accel_meas, gyro_meas, mag_meas


# ==========================================
# 2. CLASSES E FUNÇÕES AUXILIARES
# ==========================================
class PIDController:
    def __init__(self, kp, ki, kd, dt, limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.limit, self.limit)

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -self.limit, self.limit)


def get_sdre_matrices(roll, pitch, yaw, p, q, r, omega_r=0.0, dt=DT_LOOP):
    """Linearização SDC (A depende do estado) + discretização por Taylor de 2ª ordem.

    dt deve ser o período REAL entre recálculos do ganho — não um clock fixo
    assumido. Usar o dt errado aqui é o bug clássico de descasamento
    sync/async (Ad/Bd calculado para 200Hz enquanto o ciclo real é mais lento).
    """
    alpha_1, alpha_2, alpha_3 = 0.5, 0.5, 0.5
    A = np.zeros((6, 6))

    sr, cr = np.sin(roll), np.cos(roll)
    cp, tp = np.cos(pitch), np.tan(pitch)

    if abs(cp) < 1e-5:
        cp = np.sign(cp) * 1e-5
    inv_cp = 1.0 / cp

    A[0, 3] = 1.0
    A[0, 4] = sr * tp
    A[0, 5] = cr * tp
    A[1, 4] = cr
    A[1, 5] = -sr
    A[2, 4] = sr * inv_cp
    A[2, 5] = cr * inv_cp

    A[3, 4] = alpha_1 * ((Iyy - Izz) / Ixx) * r - Ir * omega_r / Ixx
    A[3, 5] = (1 - alpha_1) * ((Iyy - Izz) / Ixx) * q

    A[4, 3] = alpha_2 * ((Izz - Ixx) / Iyy) * r + Ir * omega_r / Iyy
    A[4, 5] = (1 - alpha_2) * ((Izz - Ixx) / Iyy) * p

    A[5, 3] = alpha_3 * ((Ixx - Iyy) / Izz) * q
    A[5, 4] = (1 - alpha_3) * ((Ixx - Iyy) / Izz) * p

    B = np.zeros((6, 3))
    B[3, 0] = 1.0 / Ixx
    B[4, 1] = 1.0 / Iyy
    B[5, 2] = 1.0 / Izz

    A2 = A @ A
    Ad = np.eye(6) + A * dt + A2 * (dt**2 * 0.5)
    Bd = B * dt + (A @ B) * (dt**2 * 0.5)

    return Ad, Bd


def get_rotation_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Rx = np.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
    Ry = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    Rz = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ==========================================
# 3. SIMULAÇÃO DO SISTEMA (com sensores + Madgwick)
# ==========================================
def simulate():
    """Controle síncrono, mono-thread, com timing realista do loop.

    Ordem real de execução, uma vez por iteração (sem task/thread separada):
      1) lê sensores (com atraso de fase SENSOR_DELAY_S)
      2) aplica Madgwick (dt = DT_LOOP, o período real do ciclo)
      3) aplica controle aos motores — usa o ganho K calculado no ciclo ANTERIOR
         (o SDRE deste ciclo ainda não existe: só é calculado no passo 4)
      4) calcula o novo ganho SDRE a partir da estimativa fresca deste ciclo,
         disponível para uso no ciclo SEGUINTE
    O atuador (ESC+motor+hélice) tem resposta física de 1ª ordem com constante
    de tempo MOTOR_LAG_TAU_S — a velocidade real do motor persegue o comando,
    não o acompanha instantaneamente.
    """
    from collections import deque

    time_sim = 20.0
    steps = int(time_sim / DT_LOOP)
    state = np.zeros(12)

    u_max = b_coeff * MAX_OMEGA_SQ * 4
    erro_max = 0.5
    kp_z = (u_max - g * m) / (erro_max * m)
    kd_z = 2 * np.sqrt(kp_z)

    pid_vx = PIDController(kp=0.0, ki=0.0, kd=0.0, dt=dt_pid, limit=15.0)
    pid_vy = PIDController(kp=0.0, ki=0.0, kd=0.0, dt=dt_pid, limit=15.0)
    pid_z = PIDController(kp=kp_z, ki=0.1, kd=kd_z, dt=dt_pid, limit=15.0)

    history = {
        "t": [],
        "x": [],
        "y": [],
        "z": [],
        "vx": [],
        "vy": [],
        "vz": [],
        "phi": [],
        "theta": [],
        "psi": [],
        "phi_est": [],
        "theta_est": [],
        "psi_est": [],
        "wind_x": [],
        "wind_y": [],
        "wind_z": [],
        "rpm1": [],
        "rpm2": [],
        "rpm3": [],
        "rpm4": [],
        "ax_meas": [],
        "ay_meas": [],
        "az_meas": [],
        "gx_meas": [],
        "gy_meas": [],
        "gz_meas": [],
        "mx_meas": [],
        "my_meas": [],
        "mz_meas": [],
    }

    # Ganho K em uso — sempre o calculado no ciclo ANTERIOR (ver docstring)
    K_sdre = np.zeros((3, 6))

    phi_ref, theta_ref = 0.0, 0.0
    T_ideal = m * g
    tau_ideal = np.zeros(3)

    quat = np.array([1.0, 0.0, 0.0, 0.0])
    a_cg_world_prev = np.array([0.0, 0.0, 0.0])

    # Resposta física dos motores (lag de 1ª ordem, constante de tempo MOTOR_LAG_TAU_S)
    omega_motor_actual = np.zeros(4)
    motor_lag_alpha = np.exp(-DT_LOOP / MOTOR_LAG_TAU_S)

    # Buffer de atraso do sensor com interpolação linear para atraso fracionário
    delay_steps_f = SENSOR_DELAY_S / DT_LOOP
    buf_len = max(int(np.ceil(delay_steps_f)) + 2, 2)
    _zero = (np.zeros(3), np.zeros(3), np.zeros(3))
    sensor_buf = deque([_zero] * buf_len, maxlen=buf_len)

    for step in range(steps):
        t = step * DT_LOOP
        vx, vy, vz = state[3], state[4], state[5]
        phi_t, theta_t, psi_t = state[6], state[7], state[8]
        p_t, q_t, r_t = state[9], state[10], state[11]
        z = state[2]

        Vx_ref, Vy_ref, Z_ref = 0.0, 0.0, 1.0

        # Perturbação de Vento
        w_x = np.random.normal(0.0, 0.2)
        w_y = np.random.normal(0.0, 0.2)
        w_z = np.random.normal(0.0, 0.1)

        flag_rajada = False
        if (2.0 <= t <= 3.5) and flag_rajada:
            w_x += np.random.normal(0.3, 0.1)
            w_y += np.random.normal(-0.2, 0.1)
            w_z += np.random.normal(-0.05, 0.05)

        # ====================================================
        # 1) SENSORES: leitura instantânea + buffer de atraso de fase
        # ====================================================
        R_true = get_rotation_matrix(phi_t, theta_t, psi_t)
        accel_raw, gyro_raw, mag_raw = simulate_sensors(state, a_cg_world_prev, R_true)

        # Empurra leitura atual no buffer circular
        sensor_buf.append((accel_raw, gyro_raw, mag_raw))

        # Extrai com atraso SENSOR_DELAY_S via interpolação linear entre amostras
        idx_int = int(delay_steps_f)
        idx_frac = delay_steps_f - idx_int
        n = len(sensor_buf)
        i0 = min(idx_int, n - 1)
        i1 = min(idx_int + 1, n - 1)
        s0 = sensor_buf[n - 1 - i0]
        s1 = sensor_buf[n - 1 - i1]
        accel_meas = (1.0 - idx_frac) * s0[0] + idx_frac * s1[0]
        gyro_meas = (1.0 - idx_frac) * s0[1] + idx_frac * s1[1]
        mag_meas = (1.0 - idx_frac) * s0[2] + idx_frac * s1[2]
        # ====================================================

        # 2) MADGWICK — uma vez por iteração, com dt = DT_LOOP (período real do ciclo)
        quat = madgwick_marg_update(
            quat,
            gyro_meas[0],
            gyro_meas[1],
            gyro_meas[2],
            accel_meas[0],
            accel_meas[1],
            accel_meas[2],
            mag_meas[0],
            mag_meas[1],
            mag_meas[2],
            MADGWICK_BETA,
            DT_LOOP,
        )
        phi_est, theta_est, psi_est = quat_to_euler(quat)
        # Taxas estimadas vêm direto do giroscópio (Madgwick não estima bias do giro)
        p_est, q_est, r_est = gyro_meas[0], gyro_meas[1], gyro_meas[2]

        # PID DE POSIÇÃO — laço externo mais lento, gated por dt_pid
        if step % max(1, int(round(dt_pid / DT_LOOP))) == 0:
            ux = pid_vx.update(Vx_ref - vx)
            uy = pid_vy.update(Vy_ref - vy)
            uz = pid_z.update(Z_ref - z)

            theta_ref = np.arctan2(ux, uz + g)
            phi_ref = np.arctan2(-uy * np.cos(theta_ref), uz + g)

            # Injeção de referência de Roll
            if 5.0 <= t <= 8.0:
                phi_ref = 10.0 * np.pi / 180.0
            elif 10.0 <= t <= 13.0:
                phi_ref = -10.0 * np.pi / 180.0

            T_ideal = m * (uz + g) / (np.cos(theta_ref) * np.cos(phi_ref))

        # 3) CONTROLE DE ATITUDE — u = -K*(x - x_ref)
        # K é o ganho calculado no ciclo ANTERIOR (passo 4 de uma iteração atrás);
        # o estado x_att, porém, é a estimativa FRESCA deste ciclo (passo 2 acima).
        x_att = np.array([phi_est, theta_est, psi_est, p_est, q_est, r_est])
        x_ref = np.array([phi_ref, theta_ref, 0.0, 0.0, 0.0, 0.0])
        tau_ideal = -K_sdre @ (x_att - x_ref)

        # ========================================================
        # MIXER + SATURAÇÃO DE RPM
        # ========================================================
        efforts_ideal = np.array([T_ideal, tau_ideal[0], tau_ideal[1], tau_ideal[2]])
        omega_sq_ideal = M_inv @ efforts_ideal
        omega_sq_real = np.clip(omega_sq_ideal, 0.0, MAX_OMEGA_SQ)
        omega_cmd = np.sqrt(omega_sq_real)

        # Atraso de acionamento: a velocidade REAL do motor persegue o comando
        # com lag de 1ª ordem (constante de tempo MOTOR_LAG_TAU_S ≈ 80ms)
        omega_motor_actual += (omega_cmd - omega_motor_actual) * (1.0 - motor_lag_alpha)
        rpms = omega_motor_actual * (60.0 / (2 * np.pi))

        efforts_real = M_mixer_real @ (omega_motor_actual**2)
        T_real = efforts_real[0]
        tau_real = efforts_real[1:4]
        # ========================================================

        # Dinâmica Translacional + Vento + Arrasto
        drag_accel = (-K_DRAG_TRANS / m) * np.array([vx, vy, vz])
        a_cg_world = np.array(
            [
                (T_real / m) * (np.sin(theta_t) * np.cos(phi_t)) + w_x + drag_accel[0],
                (T_real / m) * (-np.sin(phi_t)) + w_y + drag_accel[1],
                (T_real / m) * (np.cos(theta_t) * np.cos(phi_t))
                - g
                + w_z
                + drag_accel[2],
            ]
        )
        dvx, dvy, dvz = a_cg_world
        a_cg_world_prev = a_cg_world

        # Dinâmica Rotacional
        dp = tau_real[0] / Ixx + ((Iyy - Izz) / Ixx) * q_t * r_t
        dq = tau_real[1] / Iyy + ((Izz - Ixx) / Iyy) * p_t * r_t
        dr = tau_real[2] / Izz + ((Ixx - Iyy) / Izz) * p_t * q_t

        dphi = (
            p_t
            + q_t * np.sin(phi_t) * np.tan(theta_t)
            + r_t * np.cos(phi_t) * np.tan(theta_t)
        )
        dtheta = q_t * np.cos(phi_t) - r_t * np.sin(phi_t)
        dpsi = q_t * np.sin(phi_t) / np.cos(theta_t) + r_t * np.cos(phi_t) / np.cos(
            theta_t
        )

        # Integração de Euler com dt = DT_LOOP (período real do ciclo)
        state[0] += state[3] * DT_LOOP
        state[1] += state[4] * DT_LOOP
        state[2] += state[5] * DT_LOOP
        state[3] += dvx * DT_LOOP
        state[4] += dvy * DT_LOOP
        state[5] += dvz * DT_LOOP
        state[6] += dphi * DT_LOOP
        state[7] += dtheta * DT_LOOP
        state[8] += dpsi * DT_LOOP
        state[9] += dp * DT_LOOP
        state[10] += dq * DT_LOOP
        state[11] += dr * DT_LOOP

        # 4) SDRE — calculado ao FINAL do ciclo, a partir da estimativa fresca
        # deste ciclo (passo 2). Só fica disponível para o controle no ciclo
        # SEGUINTE (ver passo 3 da próxima iteração) — daí o K "atrasado".
        Ad, Bd = get_sdre_matrices(
            phi_est, theta_est, psi_est, p_est, q_est, r_est, dt=DT_LOOP
        )
        try:
            P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R_mat)
            K_sdre = np.linalg.inv(R_mat + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        except (np.linalg.LinAlgError, ValueError):
            pass  # DARE pode falhar em estado muito ill-conditioned — mantém K anterior

        # Histórico
        history["t"].append(t)
        history["x"].append(state[0])
        history["y"].append(state[1])
        history["z"].append(state[2])
        history["vx"].append(state[3])
        history["vy"].append(state[4])
        history["vz"].append(state[5])
        history["phi"].append(state[6])
        history["theta"].append(state[7])
        history["psi"].append(state[8])
        history["phi_est"].append(phi_est)
        history["theta_est"].append(theta_est)
        history["psi_est"].append(psi_est)
        history["wind_x"].append(w_x)
        history["wind_y"].append(w_y)
        history["wind_z"].append(w_z)
        history["rpm1"].append(rpms[0])
        history["rpm2"].append(rpms[1])
        history["rpm3"].append(rpms[2])
        history["rpm4"].append(rpms[3])
        history["ax_meas"].append(accel_meas[0])
        history["ay_meas"].append(accel_meas[1])
        history["az_meas"].append(accel_meas[2])
        history["gx_meas"].append(gyro_meas[0])
        history["gy_meas"].append(gyro_meas[1])
        history["gz_meas"].append(gyro_meas[2])
        history["mx_meas"].append(mag_meas[0])
        history["my_meas"].append(mag_meas[1])
        history["mz_meas"].append(mag_meas[2])

    return history


# ==========================================
# ESTUDO: SDRE SÍNCRONO vs LQR FIXO
# ==========================================
# ---- Condições iniciais extremas para cenário 'recovery' (configurável) ----
# Drone "lançado" no ar: longe do equilíbrio + alta velocidade angular.
# Aqui a linearização do LQR (em torno de zero) ignora termos como
# sin(phi)·tan(theta) e (Iyy-Izz)/Ixx·r — que dominam a dinâmica real.
IC_PHI_DEG = 80.0  # roll inicial AGRESSIVO — sin(80°)=0.985
IC_THETA_DEG = 80.0  # pitch inicial AGRESSIVO — tan(80°)=5.67, 1/cos(80°)=5.76
IC_PSI_DEG = 80.0  # yaw inicial AGRESSIVO
IC_P_DEG_S = 100.0  # roll rate — no limite p_max
IC_Q_DEG_S = 80.0  # pitch rate
IC_R_DEG_S = 200.0  # yaw rate — no limite r_max

# Sem estrangulamento de tau (pesos main.cpp são suaves → K naturalmente
# menos agressivo → NL importa mais → LQR sofre sem precisar reduzir torque)
TAU_SCALE_STUDY = 1.0
TAU_DIST_YAW = 0.0

# Passo de integração fino para a manobra agressiva isolada (independente do
# DT_LOOP do pipeline principal — aqui o objetivo é fidelidade numérica da
# dinâmica não-linear perto do gimbal lock, não timing de loop realista)
DT_ISOLATED_STUDY_S = 0.005


def simulate_attitude_isolated(
    controller_type, scenario="recovery", time_sim=3.5, seed=42
):
    """Dinâmica de atitude isolada — sem PID/Madgwick/sensores.

    Controladores comparados:
      'sync_sdre'  : K recalculado a cada passo (SDRE verdadeiro, sem atraso)
      'fixed_lqr'  : K calculado uma vez no equilíbrio (LQR clássico)

    Cenários:
      'recovery' : IC LONGE do equilíbrio + ref=zero (drone "lançado").
                   LQR linearizado em torno de zero IGNORA os termos não-lineares
                   dominantes (sin·tan, acoplamento gyro·r) → tipicamente falha.
      'steps'    : degraus simultâneos em roll/pitch/yaw (manobra agressiva).
    """
    np.random.seed(seed)
    steps = int(time_sim / DT_ISOLATED_STUDY_S)

    # Estado inicial conforme cenário
    if scenario == "recovery":
        state = np.array(
            [
                np.deg2rad(IC_PHI_DEG),
                np.deg2rad(IC_THETA_DEG),
                np.deg2rad(IC_PSI_DEG),
                np.deg2rad(IC_P_DEG_S),
                np.deg2rad(IC_Q_DEG_S),
                np.deg2rad(IC_R_DEG_S),
            ]
        )
    else:
        state = np.zeros(6)

    # No cenário recovery, aplica TAU_SCALE_STUDY (drone subatuado)
    scale = TAU_SCALE_STUDY if scenario == "recovery" else 1.0
    tau_max = np.array([max_tau_roll, max_tau_pitch, max_tau_yaw]) * scale

    # Inicializa K no equilíbrio (todos os controladores partem daqui)
    Ad, Bd = get_sdre_matrices(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dt=DT_ISOLATED_STUDY_S)
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R_mat)
    K = np.linalg.inv(R_mat + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    history = {
        "t": [],
        "phi": [],
        "theta": [],
        "psi": [],
        "p": [],
        "q": [],
        "r": [],
        "phi_ref": [],
        "theta_ref": [],
        "psi_ref": [],
        "tau_r": [],
        "tau_p": [],
        "tau_y": [],
        "K_full": [],
    }

    for step in range(steps):
        t = step * DT_ISOLATED_STUDY_S

        # Referência por cenário
        if scenario == "recovery":
            ref = np.zeros(6)  # tarefa: estabilizar em zero
        elif scenario == "steps":
            if t < 0.3:
                ref = np.zeros(6)
            elif t < 1.5:
                ref = np.array(
                    [np.deg2rad(30), np.deg2rad(20), np.deg2rad(60), 0.0, 0.0, 0.0]
                )
            elif t < 2.7:
                ref = np.array(
                    [np.deg2rad(-30), np.deg2rad(-20), np.deg2rad(-60), 0.0, 0.0, 0.0]
                )
            else:
                ref = np.zeros(6)
        else:
            ref = np.zeros(6)

        phi, theta, psi, p, q, r = state

        # Atualização de K conforme controlador
        if controller_type == "sync_sdre":
            Ad, Bd = get_sdre_matrices(phi, theta, psi, p, q, r, dt=DT_ISOLATED_STUDY_S)
            try:
                P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R_mat)
                K = np.linalg.inv(R_mat + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
            except (np.linalg.LinAlgError, ValueError):
                pass  # DARE pode falhar em estado muito ill-conditioned — mantém K anterior
        # fixed_lqr: K nunca muda

        # Controle saturado em torque máximo físico
        tau = -K @ (state - ref)
        tau = np.clip(tau, -tau_max, tau_max)

        # Dinâmica verdadeira (Euler-Newton com acoplamento giroscópico completo)
        # Disturbance permanente de yaw simula vento de cauda (só no recovery)
        tau_dist_y = TAU_DIST_YAW if scenario == "recovery" else 0.0
        dp = tau[0] / Ixx + ((Iyy - Izz) / Ixx) * q * r
        dq = tau[1] / Iyy + ((Izz - Ixx) / Iyy) * p * r
        dr = (tau[2] + tau_dist_y) / Izz + ((Ixx - Iyy) / Izz) * p * q
        # Singularidade gimbal: pitch perto de ±90° trava simulação
        cp = np.cos(theta)
        if abs(cp) < 1e-3:
            cp = np.sign(cp) * 1e-3 if cp != 0 else 1e-3
        dphi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        dtheta = q * np.cos(phi) - r * np.sin(phi)
        dpsi = q * np.sin(phi) / cp + r * np.cos(phi) / cp

        state[0] += dphi * DT_ISOLATED_STUDY_S
        state[1] += dtheta * DT_ISOLATED_STUDY_S
        state[2] += dpsi * DT_ISOLATED_STUDY_S
        state[3] += dp * DT_ISOLATED_STUDY_S
        state[4] += dq * DT_ISOLATED_STUDY_S
        state[5] += dr * DT_ISOLATED_STUDY_S

        history["t"].append(t)
        history["phi"].append(state[0])
        history["theta"].append(state[1])
        history["psi"].append(state[2])
        history["p"].append(state[3])
        history["q"].append(state[4])
        history["r"].append(state[5])
        history["phi_ref"].append(ref[0])
        history["theta_ref"].append(ref[1])
        history["psi_ref"].append(ref[2])
        history["tau_r"].append(tau[0])
        history["tau_p"].append(tau[1])
        history["tau_y"].append(tau[2])
        history["K_full"].append(K.copy())

    return history


def run_sdre_vs_lqr_study(scenario="recovery"):
    """Compara SDRE síncrono (K recalculado a cada passo) vs LQR fixo.

    Cenário padrão: 'recovery' — IC longe do equilíbrio + ref=zero.
    Regime onde a linearização do LQR em torno do equilíbrio é INVÁLIDA.

    Métrica-chave: redução do erro RMS de tracking do SDRE em relação ao LQR
    fixo (calculado uma única vez, no equilíbrio).
    """
    print("\n" + "=" * 64)
    print(" ESTUDO — SDRE síncrono vs LQR fixo")
    print(f" Cenário: {scenario}")
    print("=" * 64)
    print(f"   DT_ISOLATED_STUDY_S : {DT_ISOLATED_STUDY_S * 1000:.1f} ms")
    print(f"   tau_max (roll)      : {max_tau_roll * 1e3:.2f} mN·m")
    p_dot_max = max_tau_roll / Ixx
    print(
        f"   p_dot_max           : {p_dot_max:.0f} rad/s²  "
        f"({np.degrees(p_dot_max):.0f} °/s²)"
    )

    if scenario == "recovery":
        # Diagnóstico: quanto a A real difere da A linearizada em zero?
        A04_real = np.sin(np.deg2rad(IC_PHI_DEG)) * np.tan(np.deg2rad(IC_THETA_DEG))
        A34_real = ((Iyy - Izz) / Ixx) * np.deg2rad(IC_R_DEG_S)
        print("")
        print(f"   tau_scale      : {TAU_SCALE_STUDY*100:.0f}% (drone subatuado)")
        print(f"   tau disturbance yaw : {TAU_DIST_YAW*1e3:.2f} mN·m permanente")
        print(f"   --- Cegueira do LQR no estado inicial ---")
        print(
            f"   IC: phi={IC_PHI_DEG}° theta={IC_THETA_DEG}° psi={IC_PSI_DEG}°  "
            f"p={IC_P_DEG_S}°/s q={IC_Q_DEG_S}°/s r={IC_R_DEG_S}°/s"
        )
        print(f"   A[0,4] (sin(phi)·tan(theta))  : LQR=0.000  Real={A04_real:.3f}")
        print(f"   A[3,4] ((Iyy-Izz)/Ixx · r)    : LQR=0.000  Real={A34_real:.3f}")
    print("=" * 64)

    h_sync = simulate_attitude_isolated("sync_sdre", scenario=scenario)
    h_fixed = simulate_attitude_isolated("fixed_lqr", scenario=scenario)

    K_sync = np.array(h_sync["K_full"])
    K_fixed = np.array(h_fixed["K_full"])

    # Distância Frobenius entre os ganhos K ao longo do tempo
    d_sync_fixed = np.linalg.norm(K_sync - K_fixed, axis=(1, 2))

    # Métrica de desempenho: RMS de erro de tracking em ângulos
    def rms_err(h):
        e_phi = np.array(h["phi"]) - np.array(h["phi_ref"])
        e_theta = np.array(h["theta"]) - np.array(h["theta_ref"])
        e_psi = np.array(h["psi"]) - np.array(h["psi_ref"])
        return np.degrees(np.sqrt(np.mean(e_phi**2 + e_theta**2 + e_psi**2)))

    rms_sync = rms_err(h_sync)
    rms_fixed = rms_err(h_fixed)
    reducao_pct = (
        (rms_fixed - rms_sync) / rms_fixed * 100.0 if rms_fixed > 1e-9 else 0.0
    )

    print(f"\n   ||K_sync - K_fixed|| médio = {np.mean(d_sync_fixed):.3e}")
    print(f"   RMS erro sync_sdre = {rms_sync:7.2f}°")
    print(f"   RMS erro fixed_lqr = {rms_fixed:7.2f}°")
    print(f"   Redução de erro do SDRE vs LQR fixo = {reducao_pct:.1f}%")

    # Detecta divergência (estado explode → controlador falhou)
    def diverged(h):
        return (
            np.max(np.abs(h["phi"])) > np.pi * 2
            or np.max(np.abs(h["theta"])) > np.pi * 0.95  # gimbal lock
            or np.max(np.abs(h["p"])) > np.deg2rad(2000)
            or not np.isfinite(np.array(h["phi"])).all()
        )

    def settling_time_s(h, tol_deg=5.0):
        """Tempo até |phi|, |theta|, |psi| ficarem < tol_deg e permanecerem."""
        arr = np.degrees(
            np.sqrt(
                np.array(h["phi"]) ** 2
                + np.array(h["theta"]) ** 2
                + np.array(h["psi"]) ** 2
            )
        )
        below = arr < tol_deg
        if not below.any():
            return float("inf")
        # último cruzamento acima → 1º index abaixo daí em diante
        last_above = np.where(~below)[0]
        if len(last_above) == 0:
            return 0.0
        idx = last_above[-1] + 1
        return idx * DT_ISOLATED_STUDY_S if idx < len(arr) else float("inf")

    print(f"\n   RMS erro angular total (||[phi,theta,psi]-ref||):")
    for name, h in [("sync_sdre", h_sync), ("fixed_lqr", h_fixed)]:
        div = " [DIVERGIU]" if diverged(h) else ""
        ts = settling_time_s(h)
        ts_str = f"{ts:.2f}s" if np.isfinite(ts) else "não estabiliza"
        print(f"     {name:11s} = {rms_err(h):7.2f}°   settle(±5°)={ts_str}{div}")
    print("=" * 64 + "\n")

    # ===== GRÁFICOS =====
    t = np.array(h_sync["t"])
    colors = {"sync_sdre": "g", "fixed_lqr": "r"}
    histories = {"sync_sdre": h_sync, "fixed_lqr": h_fixed}

    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    fig.suptitle(
        f"Estudo SDRE vs LQR Fixo — Cenário '{scenario}'  "
        f"IC: φ={IC_PHI_DEG}° θ={IC_THETA_DEG}° ψ={IC_PSI_DEG}°  "
        f"p={IC_P_DEG_S}°/s q={IC_Q_DEG_S}°/s r={IC_R_DEG_S}°/s   "
        f"(redução de erro = {reducao_pct:.1f}%)",
        fontsize=11,
    )

    angles = [("phi", "Roll φ"), ("theta", "Pitch θ"), ("psi", "Yaw ψ")]
    for col, (k, lbl) in enumerate(angles):
        ax = axes[0, col]
        for name, h in histories.items():
            ax.plot(t, np.degrees(h[k]), colors[name], label=name, lw=1.4, alpha=0.85)
        ax.plot(t, np.degrees(h_sync[k + "_ref"]), "k--", lw=1, label="ref")
        ax.set_title(f"{lbl} (°)")
        ax.set_xlabel("t (s)")
        ax.grid(True)
        ax.legend(fontsize=8)

    rates = [("p", "p (roll rate)"), ("q", "q (pitch rate)"), ("r", "r (yaw rate)")]
    for col, (k, lbl) in enumerate(rates):
        ax = axes[1, col]
        for name, h in histories.items():
            ax.plot(t, np.degrees(h[k]), colors[name], label=name, lw=1.3, alpha=0.85)
        ax.set_title(f"{lbl} (°/s)")
        ax.set_xlabel("t (s)")
        ax.grid(True)
        ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(t, [np.linalg.norm(M) for M in K_sync], "g", label="||K_sync||", lw=1.4)
    ax.plot(t, [np.linalg.norm(M) for M in K_fixed], "r", label="||K_fixed||", lw=1.4)
    ax.set_title("||K||  Frobenius")
    ax.set_xlabel("t (s)")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(t, d_sync_fixed, "g", label="||K_sync − K_fixed||", lw=1.4)
    ax.set_title("Distância entre ganhos K (SDRE vs LQR fixo)")
    ax.set_xlabel("t (s)")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axes[2, 2]
    for name, h in histories.items():
        tau_mag = np.sqrt(
            np.array(h["tau_r"]) ** 2
            + np.array(h["tau_p"]) ** 2
            + np.array(h["tau_y"]) ** 2
        )
        ax.plot(t, tau_mag * 1e3, colors[name], label=name, lw=1.3, alpha=0.85)
    tau_max_norm = np.linalg.norm([max_tau_roll, max_tau_pitch, max_tau_yaw]) * 1e3
    ax.axhline(tau_max_norm, color="k", ls=":", lw=1, label="||τ_max||")
    ax.set_title("Esforço de controle ||τ|| (mN·m)")
    ax.set_xlabel("t (s)")
    ax.grid(True)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return h_sync, h_fixed


# ==========================================
# EXECUÇÃO
# ==========================================
# True -> roda o estudo comparativo SDRE vs LQR fixo (manobra de recuperação
# agressiva, dinâmica isolada) em vez da simulação principal.
RUN_SDRE_VS_LQR_STUDY = False

if RUN_SDRE_VS_LQR_STUDY:
    print("Rodando estudo: SDRE síncrono vs LQR fixo...")
    _h_sync, _h_fixed = run_sdre_vs_lqr_study()
    plt.show()
    import sys

    sys.exit(0)

print(
    f"Rodando simulação síncrona — DT_LOOP={DT_LOOP * 1000:.1f}ms "
    f"(~{1.0 / DT_LOOP:.0f}Hz), atraso de motor={MOTOR_LAG_TAU_S * 1000:.0f}ms..."
)
history = simulate()

# ==========================================
# 4. GRÁFICOS — DESEMPENHO DE CONTROLE
# ==========================================
fig1 = plt.figure(figsize=(14, 14))

plt.subplot(4, 1, 1)
plt.plot(
    history["t"], history["wind_x"], label="Vento em X", color="darkred", alpha=0.7
)
plt.plot(
    history["t"], history["wind_y"], label="Vento em Y", color="darkgreen", alpha=0.7
)
plt.plot(
    history["t"], history["wind_z"], label="Vento em Z", color="darkblue", alpha=0.7
)
plt.axvspan(2.0, 3.5, color="gray", alpha=0.2, label="Rajada Forte")
plt.title("Perturbação de Vento Injetada")
plt.ylabel("Aceleração (m/s²)")
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(history["t"], history["vx"], label="Vx Real", color="r")
plt.plot(history["t"], history["vy"], label="Vy Real", color="g")
plt.plot(history["t"], history["z"], label="Z Real (Altitude)", color="b")
plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ref Z (1m)")
plt.axvspan(2.0, 3.5, color="gray", alpha=0.2)
plt.title("Efeito do Vento nas Velocidades e Altitude")
plt.ylabel("m/s e m")
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(history["t"], np.degrees(history["phi"]), label="Roll ($\\phi$)", color="g")
plt.plot(
    history["t"], np.degrees(history["theta"]), label="Pitch ($\\theta$)", color="r"
)
plt.plot(history["t"], np.degrees(history["psi"]), label="Yaw ($\\psi$)", color="b")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.axvspan(2.0, 3.5, color="gray", alpha=0.2)
plt.title("Reação de Atitude Comandada (Graus) — Estado Real")
plt.ylabel("Ângulo (graus)")
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(history["t"], history["rpm1"], label="Motor 1 (Front-Dir CW)", color="orange")
plt.plot(history["t"], history["rpm2"], label="Motor 2 (Tras-Dir CCW)", color="purple")
plt.plot(history["t"], history["rpm3"], label="Motor 3 (Tras-Esq CW)", color="cyan")
plt.plot(
    history["t"], history["rpm4"], label="Motor 4 (Front-Esq CCW)", color="magenta"
)
plt.axhline(
    31086, color="red", linestyle="--", linewidth=2, label="Limite Físico (31.086 RPM)"
)
plt.axvspan(2.0, 3.5, color="gray", alpha=0.2)
plt.title("Atuação Física: Saturação de Rotação dos Motores")
plt.xlabel("Tempo (s)")
plt.ylabel("Rotação (RPM)")
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()


# ==========================================
# 4B. SENSORES BRUTOS (MPU6050 + QMC5883L)
# ==========================================
fig_sensors = plt.figure(figsize=(14, 12))

plt.subplot(3, 1, 1)
plt.plot(history["t"], history["ax_meas"], label="Acel X", color="r", alpha=0.7)
plt.plot(history["t"], history["ay_meas"], label="Acel Y", color="g", alpha=0.7)
plt.plot(history["t"], history["az_meas"], label="Acel Z", color="b", alpha=0.7)
plt.axhline(g, color="black", linestyle="--", linewidth=1, label="g = 9.81 m/s²")
plt.axvspan(2.0, 3.5, color="gray", alpha=0.2)
plt.title("MPU6050 — Acelerômetro (força específica, com ruído + bias)")
plt.ylabel("m/s²")
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(
    history["t"],
    np.degrees(history["gx_meas"]),
    label="Giro X (p)",
    color="r",
    alpha=0.7,
)
plt.plot(
    history["t"],
    np.degrees(history["gy_meas"]),
    label="Giro Y (q)",
    color="g",
    alpha=0.7,
)
plt.plot(
    history["t"],
    np.degrees(history["gz_meas"]),
    label="Giro Z (r)",
    color="b",
    alpha=0.7,
)
plt.axvspan(2.0, 3.5, color="gray", alpha=0.2)
plt.title("MPU6050 — Giroscópio (com ruído + bias)")
plt.ylabel("°/s")
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(history["t"], history["mx_meas"], label="Mag X", color="r", alpha=0.7)
plt.plot(history["t"], history["my_meas"], label="Mag Y", color="g", alpha=0.7)
plt.plot(history["t"], history["mz_meas"], label="Mag Z", color="b", alpha=0.7)
plt.axvspan(2.0, 3.5, color="gray", alpha=0.2)
plt.title("QMC5883L — Magnetômetro (normalizado, com ruído)")
plt.xlabel("Tempo (s)")
plt.ylabel("Campo (norm.)")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()


# ==========================================
# 4C. ESTIMATIVA MADGWICK × ESTADO REAL
# ==========================================
fig_est = plt.figure(figsize=(14, 10))

ang_pairs = [
    ("phi", "Roll ($\\phi$)"),
    ("theta", "Pitch ($\\theta$)"),
    ("psi", "Yaw ($\\psi$)"),
]
for i, (key, label) in enumerate(ang_pairs):
    plt.subplot(3, 1, i + 1)
    real = np.degrees(np.array(history[key]))
    est = np.degrees(np.array(history[key + "_est"]))
    plt.plot(history["t"], real, label=f"{label} Real", linewidth=2)
    plt.plot(
        history["t"],
        est,
        label=f"{label} Estimado (Madgwick)",
        linewidth=1.2,
        linestyle="--",
    )
    plt.plot(
        history["t"],
        est - real,
        label="Erro (Est − Real)",
        color="orange",
        alpha=0.6,
    )
    plt.axvspan(2.0, 3.5, color="gray", alpha=0.2)
    plt.title(f"{label}: Real vs Madgwick")
    plt.ylabel("Graus")
    plt.legend(loc="upper right")
    plt.grid(True)

plt.xlabel("Tempo (s)")
plt.tight_layout()


# ==========================================
# 5. ANIMAÇÃO 3D
# ==========================================
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection="3d")
ax.set_title("Visualização 3D - Drone Quadricóptero")

hx, hy, hz = np.array(history["x"]), np.array(history["y"]), np.array(history["z"])
hphi, htheta, hpsi = (
    np.array(history["phi"]),
    np.array(history["theta"]),
    np.array(history["psi"]),
)

lim = max(np.max(np.abs(hx)), np.max(np.abs(hy)), 1.0)
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

L = lim * 0.15
arm1 = np.array([L, 0, 0])
arm2 = np.array([-L, 0, 0])
arm3 = np.array([0, L, 0])
arm4 = np.array([0, -L, 0])

(line_x,) = ax.plot([], [], [], "r-", lw=4, label="Eixo X (Pitch)")
(line_y,) = ax.plot([], [], [], "b-", lw=4, label="Eixo Y (Roll)")
(path,) = ax.plot([], [], [], "k--", alpha=0.5, label="Trajetória")
ax.legend()


def update_anim(frame):
    path.set_data(hx[:frame], hy[:frame])
    path.set_3d_properties(hz[:frame])

    R = get_rotation_matrix(hphi[frame], htheta[frame], hpsi[frame])
    p_center = np.array([hx[frame], hy[frame], hz[frame]])

    p1 = p_center + R @ arm1
    p2 = p_center + R @ arm2
    p3 = p_center + R @ arm3
    p4 = p_center + R @ arm4

    line_x.set_data([p1[0], p2[0]], [p1[1], p2[1]])
    line_x.set_3d_properties([p1[2], p2[2]])

    line_y.set_data([p3[0], p4[0]], [p3[1], p4[1]])
    line_y.set_3d_properties([p3[2], p4[2]])

    return line_x, line_y, path


anim_stride = max(1, round(0.020 / DT_LOOP))  # mantém velocidade real (20ms/frame)
ani = animation.FuncAnimation(
    fig2,
    update_anim,
    frames=np.arange(0, len(hx), anim_stride),
    interval=20,
    blit=False,
)
plt.show()
