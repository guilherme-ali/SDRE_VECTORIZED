/**
 * BENCHMARK: Solvers DARE sobre trajetórias de voo determinísticas
 *
 * Substitui o passeio aleatório por perturbação uniforme (faixas mínimas em
 * torno do hover) da versão anterior deste arquivo — ver
 * G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\revisoes_consolidadas.md,
 * item 2.2 (crítica do revisor 2 do CBA 2026: "condições próximas ao pairar,
 * o que pode não estressar os métodos numéricos sob manobras agressivas" e
 * "abordagem estatística de malha aberta não captura a verdadeira trajetória
 * sequencial contínua de um drone em voo").
 *
 * Seis trajetórias determinísticas e fechadas (sem RNG — resolve também a
 * objeção de reprodutibilidade do revisor 3), definidas de uma vez só em
 * lib/Trajectories/Trajectories.h (fonte única, espelhada em
 * python/trajetorias.py) e avaliadas ponto a ponto, O(1) de memória (nada de
 * armazenar os 11538 pontos por trajetória — cada ponto é calculado por
 * avaliação direta de t-DT/t/t+DT, exceto T4, cuja conformação de 1a ordem é
 * uma recursão streamada):
 *
 *   T1_espiral    — espiral circular de raio crescente (pedido do
 *                   coautor Reginaldo Cardoso): aceleração centrípeta
 *                   omega^2*R(t) cresce linearmente, varrendo do quase-hover
 *                   ao limite prático de inclinação.
 *   T2_figura8    — Lissajous 1:2 em atitude (figura-8): inversões
 *                   periódicas de sinal nos termos cruzados de A(x).
 *   T3_chirp      — chirp linear em roll/pitch, 0,2 a 8 Hz: excita toda a
 *                   banda num único experimento; eixo x interpretável.
 *   T4_degrau_yaw — degraus agressivos (±40°) conformados por 1a ordem +
 *                   giro de guinada contínuo: pior caso do termo r e do
 *                   warm-start do método iterativo.
 *   T5_tilt_alto  — theta varrendo ±80° (sec theta = 5,8) em 15 s: isola o
 *                   efeito do ÂNGULO sobre o condicionamento, com taxas
 *                   deliberadamente baixas (~34°/s).
 *   T6_taxa_alta  — 45° a 10 Hz em roll/pitch: p,q ~2800°/s, 2,3x o máximo
 *                   observado em T1-T4; isola o efeito da TAXA, com ângulos
 *                   dentro dos 60° padrão.
 *
 * Doze métodos avaliados (seis em float + os seis pares em ponto fixo
 * Q13.18 da extensão da auditoria — ver docs/auditoria_solvers_riccati.md,
 * Seção 8): SDA, SDA_SS, ASDA, SDA_SCALED, ADDA, ITERATIVE, e os *_FIXED
 * correspondentes. VAN_DOOREN e SCHUR ficam de fora — decisão do usuário e
 * também elimina o fator de confusão apontado pelo revisor 2 (o uso da
 * biblioteca Eigen só para esses dois métodos misturava sobrecarga de
 * biblioteca com custo algorítmico na comparação).
 *
 * ITERATIVE/ITERATIVE_FIXED voltaram para esta bateria (2026-08-18) depois
 * de terem sido excluídos por custo (ver histórico em
 * docs/auditoria_solvers_riccati.md, Seção 13): o Exp. 0b
 * (test/tolerance_sweep.cpp) os avalia em pontos DECIMADOS (stride ~0,6s
 * entre amostras consecutivas), o que destrói justamente a premissa do
 * warm-start — em voo real o ciclo anterior está a 5,2 ms de distância, não
 * a 600 ms, e é essa proximidade que dá ao warm-start um ponto de partida
 * útil. Esta bateria já reaproveita a MESMA instância AutoLQR ao longo dos
 * 11538 pontos consecutivos de cada trajetória (laço abaixo), então
 * ITERATIVE/ITERATIVE_FIXED recebem warm-start genuíno de fábrica, sem
 * nenhuma configuração extra — o Exp. 0b mede o pior caso (sem vizinhança
 * temporal), esta bateria mede o caso real de voo.
 *
 * Critério de parada CASADO entre float e ponto fixo (Seção 13/14 da
 * auditoria): norma de FROBENIUS relativa (‖H_{k+1}-H_k‖_F/‖H_k‖_F — a
 * mesma norma nos dois caminhos, por decisão explícita do usuário; o
 * caminho fixed-point calcula a soma de quadrados em float a partir dos
 * valores Q13.18 convertidos, ver FixedPointQ.cpp). Tolerância e orçamento
 * re-derivados empiricamente no Exp. 0a sob este critério (ver
 * outputs/serial_tolerance_sweep.txt e python/analisa_tolerancia.py) — os
 * valores abaixo são os defaults de compilação, não necessariamente os
 * mesmos 1e-6/200 usados sob o critério máx-abs anterior. Mesmo orçamento
 * aplicado a
 * ITERATIVE/ITERATIVE_FIXED — se não bastar mesmo com warm-start, o
 * outcome=budget (não uma falha "escondida") registra isso. Configurável em
 * tempo de compilação via -DBATTERY_REL_TOL/-DBATTERY_MAX_ITERS.
 *
 * A(x), Ad, Bd, Qd, Rd são a MESMA discretização analítica esparsa que voa
 * de verdade (test/main_backup.cpp:910-1046 — o firmware real; src/main.cpp
 * está temporariamente ocupado pelo harness de host da auditoria, ver nota
 * no início da sessão), com os parâmetros físicos identificados por ensaio
 * (Ixx/Iyy/Izz/Ir) e os pesos Q/R pela regra de Bryson a partir dos limites
 * físicos reais de atitude e torque — não valores escolhidos a dedo
 * (resposta às anotações #24/#25 do Reginaldo e à crítica do revisor 2 sobre
 * Q/R fixos sem justificativa).
 *
 * Tempo medido por esp_timer_get_time() (resolução de 1 µs, monotônico,
 * ESP-IDF), uma execução por ponto — metodologia de medição documentada
 * aqui por completo (resposta ao revisor 3, "não está claramente descrito o
 * procedimento utilizado para aquisição... dos tempos").
 *
 * Saída em CSV pelo serial (921600 baud — 800.000+ linhas no total; a
 * 115200 baud a transmissão dominaria o tempo total do benchmark):
 *   PT,<traj>,<k>,<t>,<phi>,<theta>,<psi>,<p>,<q>,<r>            — 1/ponto
 *   RUN,<traj>,<k>,<metodo>,<time_us>,<iters>,<residuo_dare>,<ok> — decimado 1:5
 *   GAIN,<traj>,<k>,<metodo>,K00..K25                             — decimado 1:50
 *   SUMMARY,<traj|ALL>,<metodo>,mean_us,std_us,max_us,mean_iters,max_iters,
 *           mean_res,max_res,failures,count                       — 1/método/trajetória + geral
 * Linhas de comentário/progresso começam com '#'.
 *
 * A "falha" de um método é o próprio retorno de computeGains() (ok=false):
 * overflow/singularidade em ponto fixo, não-convergência em max_iter no
 * float — critério interno do solver, não um limiar externo arbitrário de
 * iterações (o benchmark anterior contava "iter > 10" como falha mesmo
 * quando o solver reportava sucesso; ver revisoes_consolidadas.md).
 *
 * Consumidor em Python: python/bench_trajetorias.py (referência de dupla
 * precisão via scipy.linalg.solve_discrete_are, percentis, figuras).
 */

#include <Arduino.h>
#include <AutoLQR.h>
#include <math.h>
#include <esp_timer.h>
#include "MatrixOperations.h"
#include "Trajectories.h"

namespace RiccatiBenchmark {

static const int N = 6, M = 3;

// ===== Parâmetros físicos reais (idênticos a test/main_backup.cpp) =====
const float Ixx = 42.95e-6f, Iyy = 37.77e-6f, Izz = 76.15e-6f, Ir = 1.02e-7f;
const float L_ARM = 0.060f * 0.70710678f;
const float MOTOR_B = 2.98e-8f, MOTOR_D = 0.05f * MOTOR_B;
const float MAX_RPM = 26423.0f;
const float MAX_OMEGA = (MAX_RPM * 2.0f * (float)M_PI) / 60.0f;
// s — período real do laço de controle (não os 12 ms do artigo do CBA).
// Fonte única em lib/Trajectories/Trajectories.h.
static const float DT = Trajectories::DT;

// ===== Q/R pela regra de Bryson, a partir dos limites físicos (não valores a dedo) =====
const float roll_max = 45.0f * DEG_TO_RAD, pitch_max = 45.0f * DEG_TO_RAD, yaw_max = 90.0f * DEG_TO_RAD;
const float p_max = 300.0f * DEG_TO_RAD, q_max = 300.0f * DEG_TO_RAD, r_max = 200.0f * DEG_TO_RAD;
const float Q_11 = 1.0f / (roll_max * roll_max);
const float Q_22 = 1.0f / (pitch_max * pitch_max);
const float Q_33 = 1.0f / (yaw_max * yaw_max);
const float Q_44 = 1.0f / (p_max * p_max);
const float Q_55 = 1.0f / (q_max * q_max);
const float Q_66 = 1.0f / (r_max * r_max);

const float perc_tau_max = 0.5f;
const float max_tau_roll = 2.0f * MOTOR_B * L_ARM * MAX_OMEGA * MAX_OMEGA * perc_tau_max;
const float max_tau_pitch = max_tau_roll;
const float max_tau_yaw = 4.0f * MOTOR_D * MAX_OMEGA * MAX_OMEGA * perc_tau_max;
const float R_11 = 1.0f / (max_tau_roll * max_tau_roll);
const float R_22 = 1.0f / (max_tau_pitch * max_tau_pitch);
const float R_33 = 1.0f / (max_tau_yaw * max_tau_yaw);

float Ad[N * N], Bd[N * M], Qd[N * N], Rd[M * M];

// ---------------------------------------------------------------------------
// A(x), Ad, Bd, Qd, Rd — cópia fiel da discretização analítica esparsa que
// roda de verdade no firmware (test/main_backup.cpp:910-1046), com
// omega_r=0 (não simulamos velocidade residual dos rotores nesta bateria,
// mesma simplificação do harness C1_hover em test/verify_gains_onboard.cpp)
// e escrevendo em buffers locais do namespace em vez de globais do firmware.
// ---------------------------------------------------------------------------
static void updateSystemMatrix(float roll, float pitch, float p, float q, float r) {
    const float omega_r = 0.0f;
    const float inv_Ixx = 1.0f / Ixx, inv_Iyy = 1.0f / Iyy, inv_Izz = 1.0f / Izz;
    const float Iyy_Izz_over_Ixx = (Iyy - Izz) / Ixx;
    const float Izz_Ixx_over_Iyy = (Izz - Ixx) / Iyy;
    const float Ixx_Iyy_over_Izz = (Ixx - Iyy) / Izz;
    const float Ir_over_Ixx = Ir / Ixx;
    const float Ir_over_Iyy = Ir / Iyy;
    const float dt = DT, dt2_2 = dt * dt * 0.5f;

    const float sR = sinf(roll), cR = cosf(roll);
    const float sP = sinf(pitch), cP = cosf(pitch);
    const float inv_cP = 1.0f / cP;
    const float tP = sP * inv_cP;

    const float A03 = 1.0f, A04 = sR * tP, A05 = cR * tP;
    const float A14 = cR, A15 = -sR;
    const float A24 = sR * inv_cP, A25 = cR * inv_cP;
    const float A34 = Iyy_Izz_over_Ixx * r - Ir_over_Ixx * omega_r;
    const float A43 = Ir_over_Iyy * omega_r;
    const float A45 = Izz_Ixx_over_Iyy * p;
    const float A54 = Ixx_Iyy_over_Izz * p;

    const float A2_03 = A04 * A43, A2_04 = A34 + A05 * A54, A2_05 = A04 * A45;
    const float A2_13 = A14 * A43, A2_14 = A15 * A54, A2_15 = A14 * A45;
    const float A2_23 = A24 * A43, A2_24 = A25 * A54, A2_25 = A24 * A45;
    const float A2_33 = A34 * A43, A2_35 = A34 * A45, A2_44 = A43 * A34 + A45 * A54;
    const float A2_53 = A54 * A43, A2_55 = A54 * A45;

    memset(Ad, 0, sizeof(Ad));
    Ad[0 * N + 0] = 1.0f; Ad[1 * N + 1] = 1.0f; Ad[2 * N + 2] = 1.0f;
    Ad[3 * N + 3] = 1.0f + A2_33 * dt2_2;
    Ad[4 * N + 4] = 1.0f + A2_44 * dt2_2;
    Ad[5 * N + 5] = 1.0f + A2_55 * dt2_2;
    Ad[0 * N + 3] = A03 * dt + A2_03 * dt2_2;
    Ad[0 * N + 4] = A04 * dt + A2_04 * dt2_2;
    Ad[0 * N + 5] = A05 * dt + A2_05 * dt2_2;
    Ad[1 * N + 3] = A2_13 * dt2_2;
    Ad[1 * N + 4] = A14 * dt + A2_14 * dt2_2;
    Ad[1 * N + 5] = A15 * dt + A2_15 * dt2_2;
    Ad[2 * N + 3] = A2_23 * dt2_2;
    Ad[2 * N + 4] = A24 * dt + A2_24 * dt2_2;
    Ad[2 * N + 5] = A25 * dt + A2_25 * dt2_2;
    Ad[3 * N + 4] = A34 * dt;
    Ad[3 * N + 5] = A2_35 * dt2_2;
    Ad[4 * N + 3] = A43 * dt;
    Ad[4 * N + 5] = A45 * dt;
    Ad[5 * N + 3] = A2_53 * dt2_2;
    Ad[5 * N + 4] = A54 * dt;

    memset(Bd, 0, sizeof(Bd));
    Bd[0 * M + 0] = A03 * inv_Ixx * dt2_2;
    Bd[0 * M + 1] = A04 * inv_Iyy * dt2_2;
    Bd[0 * M + 2] = A05 * inv_Izz * dt2_2;
    Bd[1 * M + 1] = A14 * inv_Iyy * dt2_2;
    Bd[1 * M + 2] = A15 * inv_Izz * dt2_2;
    Bd[2 * M + 1] = A24 * inv_Iyy * dt2_2;
    Bd[2 * M + 2] = A25 * inv_Izz * dt2_2;
    Bd[3 * M + 0] = inv_Ixx * dt;
    Bd[3 * M + 1] = A34 * inv_Iyy * dt2_2;
    Bd[4 * M + 0] = A43 * inv_Ixx * dt2_2;
    Bd[4 * M + 1] = inv_Iyy * dt;
    Bd[4 * M + 2] = A45 * inv_Izz * dt2_2;
    Bd[5 * M + 1] = A54 * inv_Iyy * dt2_2;
    Bd[5 * M + 2] = inv_Izz * dt;

    memset(Qd, 0, sizeof(Qd));
    Qd[0 * N + 0] = Q_11 * dt; Qd[1 * N + 1] = Q_22 * dt; Qd[2 * N + 2] = Q_33 * dt;
    Qd[3 * N + 3] = Q_44 * dt; Qd[4 * N + 4] = Q_55 * dt; Qd[5 * N + 5] = Q_66 * dt;
    const float q03 = Q_11 * A03 * dt2_2, q04 = Q_11 * A04 * dt2_2, q05 = Q_11 * A05 * dt2_2;
    const float q14 = Q_22 * A14 * dt2_2, q15 = Q_22 * A15 * dt2_2;
    const float q24 = Q_33 * A24 * dt2_2, q25 = Q_33 * A25 * dt2_2;
    const float q34 = (A43 * Q_55 + Q_44 * A34) * dt2_2;
    const float q45 = (A54 * Q_66 + Q_55 * A45) * dt2_2;
    Qd[0 * N + 3] = q03; Qd[3 * N + 0] = q03;
    Qd[0 * N + 4] = q04; Qd[4 * N + 0] = q04;
    Qd[0 * N + 5] = q05; Qd[5 * N + 0] = q05;
    Qd[1 * N + 4] = q14; Qd[4 * N + 1] = q14;
    Qd[1 * N + 5] = q15; Qd[5 * N + 1] = q15;
    Qd[2 * N + 4] = q24; Qd[4 * N + 2] = q24;
    Qd[2 * N + 5] = q25; Qd[5 * N + 2] = q25;
    Qd[3 * N + 4] = q34; Qd[4 * N + 3] = q34;
    Qd[4 * N + 5] = q45; Qd[5 * N + 4] = q45;

    const float dt3_over_3 = dt * dt * dt / 3.0f;
    memset(Rd, 0, sizeof(Rd));
    Rd[0 * M + 0] = R_11 * dt + (Q_44 * inv_Ixx * inv_Ixx) * dt3_over_3;
    Rd[1 * M + 1] = R_22 * dt + (Q_55 * inv_Iyy * inv_Iyy) * dt3_over_3;
    Rd[2 * M + 2] = R_33 * dt + (Q_66 * inv_Izz * inv_Izz) * dt3_over_3;
}

// ---------------------------------------------------------------------------
// Trajetórias — agora vindas de lib/Trajectories/Trajectories.h (fonte única
// compartilhada com test/tolerance_sweep.cpp, test/gamma_sweep.cpp,
// test/sweep_qr.cpp e python/trajetorias.py). Eram ~140 linhas duplicadas
// aqui; a recursão exata de T4 deste arquivo virou a versão canônica do
// header. Seis trajetórias: T1-T4 (inalteradas) + T5_tilt_alto e
// T6_taxa_alta.
// ---------------------------------------------------------------------------
static const int N_POINTS = Trajectories::N_POINTS_FULL; // 11538
static const int N_TRAJ = Trajectories::N_TRAJ;          // 6

// ITERATIVE/ITERATIVE_FIXED de volta (ver comentário no cabeçalho do
// arquivo) — pontos consecutivos desta bateria dão warm-start genuíno,
// diferente da amostragem decimada do Exp. 0b.
static const int N_METHODS = 12;
static const char* METHODS[N_METHODS] = {
    "SDA", "SDA_SS", "ASDA", "SDA_SCALED", "ADDA", "ITERATIVE",
    "SDA_FIXED", "SDA_SS_FIXED", "ASDA_FIXED", "SDA_SCALED_FIXED", "ADDA_FIXED", "ITERATIVE_FIXED"
};

// ---------------------------------------------------------------------------
// Estatísticas por método (Welford, evita overflow/perda de precisão em
// séries longas).
//
// ANTES desta revisão, "falha" era o retorno booleano de computeGains(), e
// add() descartava a amostra inteira (return early) quando ok==false — as
// médias/desvios eram calculados só sobre os sucessos, não sobre as
// n_total tentativas. Isso enviesava especialmente o ITERATIVE float: seus
// 44135 µs médios (bateria v2) eram a média sobre os 26497 sucessos, não
// dos 46152 pontos, escondendo que 42,6% do tempo o método não converge —
// não "falhou" numericamente, apenas excedeu o orçamento de iterações (ver
// docs/auditoria_solvers_riccati.md, Seção 13, e AutoLQR::SolveOutcome).
//
// Agora TODA execução é registrada (n_total sempre incrementa), e o
// desfecho é reportado nas 3 categorias de AutoLQR::SolveOutcome:
// converged / budget / breakdown — nunca somadas como "falha" genérica.
// Os momentos (mean/M2/max) de tempo/iterações/resíduo são acumulados
// apenas sobre as execuções convergidas, porque tempo e resíduo de uma
// execução censurada por orçamento não são comparáveis ao de uma
// convergida (ela para no teto, não porque "terminou").
// ---------------------------------------------------------------------------
struct MethodStats {
    int n_total = 0;
    int n_converged = 0;
    int n_budget = 0;
    int n_breakdown = 0;
    double mean_time = 0, M2_time = 0, max_time = 0;
    double mean_iters = 0, M2_iters = 0, max_iters = 0;
    double mean_res = 0, M2_res = 0, max_res = 0;

    void add(double time_us, double iters, double res, AutoLQR::SolveOutcome outcome) {
        n_total++;
        if (outcome == AutoLQR::SolveOutcome::Budget) { n_budget++; return; }
        if (outcome == AutoLQR::SolveOutcome::Breakdown) { n_breakdown++; return; }
        n_converged++;
        if (time_us > max_time) max_time = time_us;
        if (iters > max_iters) max_iters = iters;
        if (res > max_res) max_res = res;
        auto upd = [&](double val, double& mean, double& M2) {
            double delta = val - mean;
            mean += delta / n_converged;
            double delta2 = val - mean;
            M2 += delta * delta2;
        };
        upd(time_us, mean_time, M2_time);
        upd(iters, mean_iters, M2_iters);
        upd(res, mean_res, M2_res);
    }
    double std_time() const { return n_converged > 0 ? sqrt(M2_time / n_converged) : 0.0; }
};

AutoLQR lqr[N_METHODS] = {
    AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M),
    AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M), AutoLQR(N, M),
};

MethodStats statsTraj[N_TRAJ][N_METHODS];
MethodStats statsAll[N_METHODS];

static void printSummaryLine(const char* trajLabel, int m, const MethodStats& s) {
    Serial.printf("SUMMARY,%s,%s,%.2f,%.2f,%.2f,%.3f,%.0f,%.6e,%.6e,%d,%d,%d,%d\n",
                  trajLabel, METHODS[m],
                  s.mean_time, s.std_time(), s.max_time,
                  s.mean_iters, s.max_iters,
                  s.mean_res, s.max_res,
                  s.n_converged, s.n_budget, s.n_breakdown, s.n_total);
}

// Critério de parada unificado entre float e fixed-point (ver AutoLQR::
// setStoppingCriterion), configurável em tempo de compilação para poder
// reflashar a mesma bateria com a tolerância casada escolhida no Exp. 0
// (varredura de tolerância, test/tolerance_sweep.cpp) sem editar código —
// ver docs/auditoria_solvers_riccati.md, Seção 13.
// τ=1e-3 — medido em test/tolerance_sweep.cpp (Exp. 0) como a tolerância
// mais frouxa que já satura a acurácia da família de duplicação, com
// margem de 17,8× sobre o piso de quantização do Q13.18 (ver
// docs/auditoria_solvers_riccati.md, Seção 15.2). Era 1e-6 (inalcançável
// abaixo do piso — o ITERATIVE_FIXED só "convergia" ali por coincidência
// de ponto fixo bit a bit, 9,4% dos casos).
#ifndef BATTERY_REL_TOL
#define BATTERY_REL_TOL 1e-3f
#endif
#ifndef BATTERY_MAX_ITERS
#define BATTERY_MAX_ITERS 200
#endif

void run() {
    Serial.begin(921600); // 800k+ linhas no total — 115200 dominaria o tempo total
    unsigned long t_serial = millis();
    while (!Serial && millis() - t_serial < 3000) {}
    delay(1500);

    for (int m = 0; m < N_METHODS; m++) {
        lqr[m].setStoppingCriterion(BATTERY_REL_TOL, BATTERY_MAX_ITERS);
    }

    Serial.println("# BENCHMARK: solvers DARE sobre trajetorias de voo deterministicas");
    Serial.println("# PT,traj,k,t,phi,theta,psi,p,q,r");
    Serial.println("# RUN,traj,k,metodo,time_us,iters,residuo_dare,outcome  (decimado 1:5)");
    Serial.println("# outcome: 0=converged 1=budget(censurado por orcamento, NAO e falha numerica) 2=breakdown(overflow/singular)");
    Serial.println("# GAIN,traj,k,metodo,K00..K25  (decimado 1:50)");
    Serial.println("# SUMMARY,traj_ou_ALL,metodo,mean_us,std_us,max_us,mean_iters,max_iters,mean_res,max_res,n_converged,n_budget,n_breakdown,n_total");
    Serial.printf("# DT=%.6f N_POINTS=%d N_TRAJ=%d N_METHODS=%d REL_TOL=%g MAX_ITERS=%d timer=esp_timer_get_time(us)\n",
                  DT, N_POINTS, N_TRAJ, N_METHODS, (double)BATTERY_REL_TOL, BATTERY_MAX_ITERS);

    unsigned long global_t0 = millis();

    for (int traj = 0; traj < N_TRAJ; traj++) {
        if (traj == (int)Trajectories::T4_DEGRAU_YAW) Trajectories::resetT4();

        for (int k = 0; k < N_POINTS; k++) {
            float t, phi, theta, psi, p, q, r;
            Trajectories::trajectoryPoint((Trajectories::TrajId)traj, k, t, phi, theta, psi, p, q, r);
            updateSystemMatrix(phi, theta, p, q, r);

            Serial.printf("PT,%s,%d,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                          Trajectories::TRAJ_NAMES[traj], k, t, phi, theta, psi, p, q, r);

            bool logRun = (k % 5 == 0);
            bool logGain = (k % 50 == 0);

            for (int m = 0; m < N_METHODS; m++) {
                lqr[m].setStateMatrix(Ad);
                lqr[m].setInputMatrix(Bd);
                lqr[m].setCostMatrices(Qd, Rd);

                int64_t t0 = esp_timer_get_time();
                lqr[m].computeGains(METHODS[m]);
                int64_t dt_us = esp_timer_get_time() - t0;
                int iters = lqr[m].getLastIterations();
                float resid = lqr[m].getLastResidual();
                AutoLQR::SolveOutcome outcome = lqr[m].getLastOutcome();

                statsTraj[traj][m].add((double)dt_us, (double)iters, (double)resid, outcome);
                statsAll[m].add((double)dt_us, (double)iters, (double)resid, outcome);

                if (logRun) {
                    Serial.printf("RUN,%s,%d,%s,%lld,%d,%.6e,%d\n",
                                  Trajectories::TRAJ_NAMES[traj], k, METHODS[m], (long long)dt_us, iters, resid,
                                  (int)outcome);
                }
                if (logGain) {
                    float K[M * N];
                    if (lqr[m].exportGains(K)) {
                        Serial.printf("GAIN,%s,%d,%s", Trajectories::TRAJ_NAMES[traj], k, METHODS[m]);
                        for (int i = 0; i < M * N; i++) Serial.printf(",%.6g", K[i]);
                        Serial.println();
                    }
                }
            }

            yield(); // ponto protegido para alimentar o watchdog sem sujar a medição

            if ((k + 1) % 2000 == 0) {
                unsigned long elapsed_s = (millis() - global_t0) / 1000;
                Serial.printf("# progresso traj=%s k=%d/%d elapsed=%lus\n",
                              Trajectories::TRAJ_NAMES[traj], k + 1, N_POINTS, elapsed_s);
            }
        }

        Serial.println("# ---- fim da trajetoria, resumo ----");
        for (int m = 0; m < N_METHODS; m++) {
            printSummaryLine(Trajectories::TRAJ_NAMES[traj], m, statsTraj[traj][m]);
        }
    }

    Serial.println("# ---- resumo geral (todas as trajetorias) ----");
    for (int m = 0; m < N_METHODS; m++) {
        printSummaryLine("ALL", m, statsAll[m]);
    }

    unsigned long total_s = (millis() - global_t0) / 1000;
    Serial.printf("# FIM DO BENCHMARK — tempo total decorrido: %lu s\n", total_s);
    Serial.println("\nReinicie o ESP32 para rodar novamente ou desative o modo benchmark.");
}

} // namespace RiccatiBenchmark

void setup() {
    RiccatiBenchmark::run();
}

void loop() {
    delay(5000);
}
