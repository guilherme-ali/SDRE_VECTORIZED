// Driver de verificação dos solvers de Riccati da lib/AUTOLQR — roda no host,
// sem alterar nenhuma linha de lib/AUTOLQR. Objetivo: medir o resíduo REAL da
// DARE (não a métrica interna ‖ΔH‖/‖H‖ que os solvers usam como critério de
// parada), a simetria e definitude de P, e a estabilidade de malha fechada,
// para cada método × caso de teste. Ver plano em
// C:\Users\guilh\.claude\plans\nesse-repositorio-ha-diversos-tingly-hopcroft.md
//
// Também exporta cada caso (A,B,Q,R) para outputs/cases/, para que
// python/verifica_solvers.py rode o mesmo caso contra o oráculo scipy e
// contra um espelho float64 das fórmulas do C++.

#include "AutoLQR.h"
#include "MatrixOperations.h"
#include <Eigen/Eigen>

#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <filesystem>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::MatrixXf;

namespace fs = std::filesystem;

static const double DEG2RAD = M_PI / 180.0;

// ============================================================================
// Modelo físico do quadrotor (espelha src/main.cpp e test/data_benchmark_solvers.cpp)
// ============================================================================

struct ContinuousModel {
    std::string name;
    Eigen::MatrixXf A, B, Qc, Rc; // contínuas, float32 (mesma precisão da produção)
};

// A[6x6], B[6x3] esparsos da cinemática de atitude + dinâmica de Euler,
// idênticos à updateSystemMatrix() de src/main.cpp:910-955.
static ContinuousModel buildQuadrotorModel(const std::string& name,
                                            float Ixx, float Iyy, float Izz, float Ir,
                                            float roll, float pitch, float yaw,
                                            float p, float q, float r, float omega_r) {
    const float inv_Ixx = 1.0f / Ixx;
    const float inv_Iyy = 1.0f / Iyy;
    const float inv_Izz = 1.0f / Izz;
    const float Iyy_Izz_over_Ixx = (Iyy - Izz) / Ixx;
    const float Izz_Ixx_over_Iyy = (Izz - Ixx) / Iyy;
    const float Ixx_Iyy_over_Izz = (Ixx - Iyy) / Izz;
    const float Ir_over_Ixx = Ir / Ixx;
    const float Ir_over_Iyy = Ir / Iyy;

    const float sR = sinf(roll), cR = cosf(roll);
    const float sP = sinf(pitch), cP = cosf(pitch);
    const float inv_cP = 1.0f / cP;
    const float tP = sP * inv_cP;

    const float A03 = 1.0f;
    const float A04 = sR * tP;
    const float A05 = cR * tP;
    const float A14 = cR;
    const float A15 = -sR;
    const float A24 = sR * inv_cP;
    const float A25 = cR * inv_cP;
    const float A34 = Iyy_Izz_over_Ixx * r - Ir_over_Ixx * omega_r;
    const float A43 = Ir_over_Iyy * omega_r;
    const float A45 = Izz_Ixx_over_Iyy * p;
    const float A54 = Ixx_Iyy_over_Izz * p;

    ContinuousModel m;
    m.name = name;
    m.A = MatrixXf::Zero(6, 6);
    m.A(0,3)=A03; m.A(0,4)=A04; m.A(0,5)=A05;
    m.A(1,4)=A14; m.A(1,5)=A15;
    m.A(2,4)=A24; m.A(2,5)=A25;
    m.A(3,4)=A34;
    m.A(4,3)=A43; m.A(4,5)=A45;
    m.A(5,4)=A54;

    m.B = MatrixXf::Zero(6, 3);
    m.B(3,0) = inv_Ixx;
    m.B(4,1) = inv_Iyy;
    m.B(5,2) = inv_Izz;

    (void)yaw; // yaw não entra em A/B (só integra o heading, não realimenta a dinâmica linearizada)

    m.Qc = MatrixXf::Zero(6, 6);
    m.Rc = MatrixXf::Zero(3, 3);
    return m;
}

// Q,R pela regra de Bryson, idêntica a src/main.cpp:111-148.
static void applyBrysonCost(ContinuousModel& m, float MOTOR_B, float MOTOR_D,
                             float L_ARM, float MAX_OMEGA) {
    const float roll_max  = 45.0f * (float)DEG2RAD;
    const float pitch_max = 45.0f * (float)DEG2RAD;
    const float yaw_max   = 90.0f * (float)DEG2RAD;
    const float p_max     = 300.0f * (float)DEG2RAD;
    const float q_max     = 300.0f * (float)DEG2RAD;
    const float r_max     = 200.0f * (float)DEG2RAD;

    m.Qc(0,0) = 1.0f/(roll_max*roll_max);
    m.Qc(1,1) = 1.0f/(pitch_max*pitch_max);
    m.Qc(2,2) = 1.0f/(yaw_max*yaw_max);
    m.Qc(3,3) = 1.0f/(p_max*p_max);
    m.Qc(4,4) = 1.0f/(q_max*q_max);
    m.Qc(5,5) = 1.0f/(r_max*r_max);

    const float perc = 0.5f;
    const float max_tau_roll  = 2.0f*MOTOR_B*L_ARM*MAX_OMEGA*MAX_OMEGA*perc;
    const float max_tau_pitch = max_tau_roll;
    const float max_tau_yaw   = 4.0f*MOTOR_D*MAX_OMEGA*MAX_OMEGA*perc;

    m.Rc(0,0) = 1.0f/(max_tau_roll*max_tau_roll);
    m.Rc(1,1) = 1.0f/(max_tau_pitch*max_tau_pitch);
    m.Rc(2,2) = 1.0f/(max_tau_yaw*max_tau_yaw);
}

// ============================================================================
// Discretização (Taylor 2a ordem, ZOH aproximado) — matematicamente idêntica
// às fórmulas esparsas de updateSystemMatrix(), só que densa via Eigen.
// ============================================================================

struct Case {
    std::string name;
    int n = 0, m = 0;
    std::vector<float> A, B, Q, R; // row-major flatten, o formato que AutoLQR espera
};

static std::vector<float> flattenRowMajor(const MatrixXf& M) {
    std::vector<float> out((size_t)M.rows() * M.cols());
    for (int i = 0; i < M.rows(); i++)
        for (int j = 0; j < M.cols(); j++)
            out[(size_t)i*M.cols()+j] = M(i,j);
    return out;
}

static Case discretizeAndPack(const std::string& name, const ContinuousModel& mdl,
                               float dt, bool discretizeCost) {
    const int n = (int)mdl.A.rows();
    MatrixXf I = MatrixXf::Identity(n, n);
    MatrixXf Ad = I + mdl.A*dt + mdl.A*mdl.A*(dt*dt*0.5f);
    MatrixXf Bd = mdl.B*dt + mdl.A*mdl.B*(dt*dt*0.5f);

    MatrixXf Qd, Rd;
    if (discretizeCost) {
        Qd = mdl.Qc*dt + (mdl.A.transpose()*mdl.Qc + mdl.Qc*mdl.A)*(dt*dt*0.5f);
        Rd = mdl.Rc*dt + (mdl.B.transpose()*mdl.Qc*mdl.B)*(dt*dt*dt/3.0f);
    } else {
        Qd = mdl.Qc;
        Rd = mdl.Rc;
    }

    Case c;
    c.name = name;
    c.n = n;
    c.m = (int)mdl.B.cols();
    c.A = flattenRowMajor(Ad);
    c.B = flattenRowMajor(Bd);
    c.Q = flattenRowMajor(Qd);
    c.R = flattenRowMajor(Rd);
    return c;
}

// ============================================================================
// C1 — hover nominal (src/main.cpp: Ixx/Iyy/Izz/Ir reais, dt=6.0ms, Bryson Q/R)
// ============================================================================
static Case buildC1Hover() {
    const float Ixx=42.95e-6f, Iyy=37.77e-6f, Izz=76.15e-6f, Ir=1.02e-7f;
    const float L_ARM = 0.060f*0.70710678f;
    const float MOTOR_B=2.98e-8f, MOTOR_D=0.05f*MOTOR_B;
    const float MAX_RPM=26423.0f;
    const float MAX_OMEGA=(MAX_RPM*2.0f*(float)M_PI)/60.0f;
    const float dt = 0.006f;

    ContinuousModel mdl = buildQuadrotorModel("C1_hover", Ixx,Iyy,Izz,Ir, 0,0,0,0,0,0,0);
    applyBrysonCost(mdl, MOTOR_B, MOTOR_D, L_ARM, MAX_OMEGA);
    return discretizeAndPack("C1_hover", mdl, dt, /*discretizeCost=*/true);
}

// ============================================================================
// C2 — réplica do benchmark do README (test/data_benchmark_solvers.cpp):
// dt=12ms, Q=diag(100,100,100,1,1,1), R=I, SEM discretizar custo, inércia
// log-uniforme aleatória (mesma distribuição do benchmark original).
// ============================================================================
static std::vector<Case> buildC2BenchmarkReplica(int count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<Case> cases;

    for (int k = 0; k < count; k++) {
        float log_scale = -6.0f + (5.0f) * u01(rng); // log10 entre 1e-6 e 1e-1
        float I_base = powf(10.0f, log_scale);
        float Ixx = I_base * (0.9f + 0.2f*u01(rng));
        float Iyy = I_base * (0.9f + 0.2f*u01(rng));
        float I_lat_avg = (Ixx+Iyy)/2.0f;
        float Izz = I_lat_avg * (1.5f + 0.5f*u01(rng));
        const float Ir = 1.0e-9f;

        float roll  = (u01(rng)*2.0f-1.0f) * 0.5f;
        float pitch = (u01(rng)*2.0f-1.0f) * 0.5f;
        float p = (u01(rng)*2.0f-1.0f) * 2.0f;
        float q = (u01(rng)*2.0f-1.0f) * 2.0f;
        float r = (u01(rng)*2.0f-1.0f) * 2.0f;

        ContinuousModel mdl = buildQuadrotorModel("C2_bench_" + std::to_string(k),
                                                    Ixx,Iyy,Izz,Ir, roll,pitch,0,p,q,r,0);
        mdl.Qc = MatrixXf::Zero(6,6);
        mdl.Qc.diagonal() << 100,100,100,1,1,1;
        mdl.Rc = MatrixXf::Identity(3,3);

        cases.push_back(discretizeAndPack("C2_bench_" + std::to_string(k), mdl, 0.012f,
                                           /*discretizeCost=*/false));
    }
    return cases;
}

// ============================================================================
// C3 — varredura de atitude (random walk, espelha o gerador de
// test/benchmark_solvers.cpp:300-318, reduzido para N pontos por custo de host)
// ============================================================================
static std::vector<Case> buildC3AttitudeSweep(int count, unsigned seed) {
    const float Ixx=42.95e-6f, Iyy=37.77e-6f, Izz=76.15e-6f, Ir=1.02e-7f;
    const float L_ARM = 0.060f*0.70710678f;
    const float MOTOR_B=2.98e-8f, MOTOR_D=0.05f*MOTOR_B;
    const float MAX_RPM=26423.0f;
    const float MAX_OMEGA=(MAX_RPM*2.0f*(float)M_PI)/60.0f;
    const float dt = 0.006f;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> step(-0.05f, 0.05f);
    float roll=0, pitch=0, p=0, q=0, r=0;

    std::vector<Case> cases;
    for (int k = 0; k < count; k++) {
        roll  = std::clamp(roll  + step(rng), -0.78f, 0.78f); // +-45 deg
        pitch = std::clamp(pitch + step(rng), -0.78f, 0.78f);
        p = std::clamp(p + step(rng)*5.0f, -5.0f, 5.0f);
        q = std::clamp(q + step(rng)*5.0f, -5.0f, 5.0f);
        r = std::clamp(r + step(rng)*5.0f, -5.0f, 5.0f);

        ContinuousModel mdl = buildQuadrotorModel("C3_attitude_" + std::to_string(k),
                                                    Ixx,Iyy,Izz,Ir, roll,pitch,0,p,q,r,0);
        applyBrysonCost(mdl, MOTOR_B, MOTOR_D, L_ARM, MAX_OMEGA);
        cases.push_back(discretizeAndPack("C3_attitude_" + std::to_string(k), mdl, dt, true));
    }
    return cases;
}

// ============================================================================
// C4 — mal-condicionado: dt muito pequeno (autovalor dominante -> 1, o caso
// lambda~=0.987 do diário técnico) e R quase nulo.
// ============================================================================
static std::vector<Case> buildC4IllConditioned() {
    const float Ixx=42.95e-6f, Iyy=37.77e-6f, Izz=76.15e-6f, Ir=1.02e-7f;
    const float L_ARM = 0.060f*0.70710678f;
    const float MOTOR_B=2.98e-8f, MOTOR_D=0.05f*MOTOR_B;
    const float MAX_RPM=26423.0f;
    const float MAX_OMEGA=(MAX_RPM*2.0f*(float)M_PI)/60.0f;

    std::vector<Case> cases;

    // C4a: dt minusculo -> Ad ~ I, autovalores no pencil colados em 1
    {
        ContinuousModel mdl = buildQuadrotorModel("C4a_dt_tiny", Ixx,Iyy,Izz,Ir, 0.1f,0.1f,0,0.5f,0.5f,0.5f,0);
        applyBrysonCost(mdl, MOTOR_B, MOTOR_D, L_ARM, MAX_OMEGA);
        cases.push_back(discretizeAndPack("C4a_dt_tiny", mdl, 0.0005f, true));
    }
    // C4b: R quase nulo (custo de controle desprezivel -> ganho agressivo)
    {
        ContinuousModel mdl = buildQuadrotorModel("C4b_R_small", Ixx,Iyy,Izz,Ir, 0,0,0,0,0,0,0);
        applyBrysonCost(mdl, MOTOR_B, MOTOR_D, L_ARM, MAX_OMEGA);
        mdl.Rc *= 1e-4f;
        cases.push_back(discretizeAndPack("C4b_R_small", mdl, 0.006f, true));
    }
    // C4c: Q quase nulo (sistema quase sem custo de estado)
    {
        ContinuousModel mdl = buildQuadrotorModel("C4c_Q_small", Ixx,Iyy,Izz,Ir, 0,0,0,0,0,0,0);
        applyBrysonCost(mdl, MOTOR_B, MOTOR_D, L_ARM, MAX_OMEGA);
        mdl.Qc *= 1e-4f;
        cases.push_back(discretizeAndPack("C4c_Q_small", mdl, 0.006f, true));
    }
    return cases;
}

// ============================================================================
// C5 — escalas de linha desiguais em A (expõe SDA_Scaled; com D=I o bug se
// esconde). Multiplica as linhas de estado angular por fatores bem diferentes
// das linhas de taxa, simulando unidades mal normalizadas.
// ============================================================================
static Case buildC5UnequalScale() {
    const float Ixx=42.95e-6f, Iyy=37.77e-6f, Izz=76.15e-6f, Ir=1.02e-7f;
    const float L_ARM = 0.060f*0.70710678f;
    const float MOTOR_B=2.98e-8f, MOTOR_D=0.05f*MOTOR_B;
    const float MAX_RPM=26423.0f;
    const float MAX_OMEGA=(MAX_RPM*2.0f*(float)M_PI)/60.0f;

    ContinuousModel mdl = buildQuadrotorModel("C5_unequal_scale", Ixx,Iyy,Izz,Ir, 0.3f,0.2f,0,0,0,0,0);
    applyBrysonCost(mdl, MOTOR_B, MOTOR_D, L_ARM, MAX_OMEGA);

    // Similaridade x' = Dx: A' = D A D^-1, B' = D B, Q' = D^-T Q D^-1.
    // D = diag(1000,1000,1000,1,1,1) -- estados angulares em "milirad", taxas em rad/s.
    MatrixXf D = MatrixXf::Identity(6,6);
    D(0,0)=D(1,1)=D(2,2)=1000.0f;
    MatrixXf Dinv = D.inverse();

    mdl.A = D * mdl.A * Dinv;
    mdl.B = D * mdl.B;
    mdl.Qc = Dinv.transpose() * mdl.Qc * Dinv;

    return discretizeAndPack("C5_unequal_scale", mdl, 0.006f, true);
}

// ============================================================================
// C6 — contraexemplos escalares/2x2 derivados na auditoria (regressão barata
// e determinística; ver plano, seção "Evidência já estabelecida").
// ============================================================================
static std::vector<Case> buildC6ScalarCounterexamples() {
    std::vector<Case> cases;
    // C6a: a=0.5, g=1 (b=1,r=1), q=1 -- expõe o SDA-SS (erro ~9.3% previsto)
    {
        Case c; c.name="C6a_scalar_ss"; c.n=1; c.m=1;
        c.A={0.5f}; c.B={1.0f}; c.Q={1.0f}; c.R={1.0f};
        cases.push_back(c);
    }
    // C6b: a=0.5, g=4 (b=2,r=1), q=1 -- expõe o ASDA (erro ~7.8% previsto)
    {
        Case c; c.name="C6b_scalar_asda"; c.n=1; c.m=1;
        c.A={0.5f}; c.B={2.0f}; c.Q={1.0f}; c.R={1.0f};
        cases.push_back(c);
    }
    // C6c: 2x2 com escalas de linha desiguais -- expõe o SDA_Scaled
    {
        Case c; c.name="C6c_2x2_scaled"; c.n=2; c.m=1;
        c.A = {0.9f, 1000.0f,
               0.0f, 0.8f};
        c.B = {0.001f, 1.0f};
        c.Q = {1000.0f, 0.0f,
               0.0f,    1.0f};
        c.R = {1.0f};
        cases.push_back(c);
    }
    return cases;
}

// ============================================================================
// Métricas
// ============================================================================

static MatrixXd toEigenD(const std::vector<float>& v, int rows, int cols) {
    MatrixXd M(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            M(i,j) = (double)v[(size_t)i*cols+j];
    return M;
}
static MatrixXd toEigenD(const float* v, int rows, int cols) {
    MatrixXd M(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            M(i,j) = (double)v[(size_t)i*cols+j];
    return M;
}

struct Metrics {
    std::string case_name, method;
    bool ok = false;
    int iterations = -1;
    double lib_residual = -1.0;   // getLastResidual() -- criterio interno (deltaH/H)
    double dare_residual = -1.0;  // resíduo REAL da DARE, calculado aqui em double
    double symmetry_err = -1.0;
    double lambda_min_P = std::nan("");
    double rho_closed_loop = std::nan("");
    double time_us = -1.0;
};

static Metrics evaluate(const Case& c, const std::string& method) {
    Metrics met;
    met.case_name = c.name;
    met.method = method;

    AutoLQR lqr(c.n, c.m);
    lqr.setStateMatrix(c.A.data());
    lqr.setInputMatrix(c.B.data());
    lqr.setCostMatrices(c.Q.data(), c.R.data());

    auto t0 = std::chrono::high_resolution_clock::now();
    met.ok = lqr.computeGains(method.c_str());
    auto t1 = std::chrono::high_resolution_clock::now();
    met.time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    met.iterations = lqr.getLastIterations();
    met.lib_residual = lqr.getLastResidual();

    MatrixXd Ad = toEigenD(c.A, c.n, c.n);
    MatrixXd Bd = toEigenD(c.B, c.n, c.m);
    MatrixXd Qd = toEigenD(c.Q, c.n, c.n);
    MatrixXd Rd = toEigenD(c.R, c.m, c.m);

    const float* Pf = lqr.getRicattiSolution();
    if (Pf == nullptr) return met;
    MatrixXd P = toEigenD(Pf, c.n, c.n);

    MatrixXd S = Rd + Bd.transpose()*P*Bd;
    MatrixXd Sinv;
    bool sOk = std::abs(S.determinant()) > 1e-30;
    if (c.m > 4 || !sOk) {
        // determinante pode subflow para m maior; usa solve robusto de qualquer forma
        Sinv = S.completeOrthogonalDecomposition().pseudoInverse();
    } else {
        Sinv = S.inverse();
    }

    MatrixXd resid = Ad.transpose()*P*Ad - P
                    - Ad.transpose()*P*Bd*Sinv*Bd.transpose()*P*Ad + Qd;
    double qNorm = std::max(1e-30, Qd.norm());
    met.dare_residual = resid.norm() / qNorm;

    double pNorm = std::max(1e-30, P.norm());
    met.symmetry_err = (P - P.transpose()).norm() / pNorm;

    MatrixXd Psym = 0.5*(P + P.transpose());
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(Psym);
    met.lambda_min_P = es.eigenvalues().minCoeff();

    std::vector<float> Kf((size_t)c.m*c.n, 0.0f);
    lqr.exportGains(Kf.data());
    MatrixXd K = toEigenD(Kf, c.m, c.n);
    MatrixXd closed = Ad - Bd*K;
    Eigen::EigenSolver<MatrixXd> ces(closed);
    met.rho_closed_loop = ces.eigenvalues().cwiseAbs().maxCoeff();

    return met;
}

// ============================================================================
// I/O
// ============================================================================

static void exportCaseCsv(const fs::path& dir, const Case& c) {
    std::ofstream f(dir / (c.name + ".csv"));
    f << "n," << c.n << "\n";
    f << "m," << c.m << "\n";
    auto dump = [&](const char* label, const std::vector<float>& v, int rows, int cols){
        f << label << "\n";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                f << v[(size_t)i*cols+j];
                if (j+1<cols) f << ",";
            }
            f << "\n";
        }
    };
    dump("A", c.A, c.n, c.n);
    dump("B", c.B, c.n, c.m);
    dump("Q", c.Q, c.n, c.n);
    dump("R", c.R, c.m, c.m);
}

int main() {
    fs::create_directories("outputs/cases");

    std::vector<Case> cases;
    cases.push_back(buildC1Hover());
    for (auto& c : buildC2BenchmarkReplica(20, 42)) cases.push_back(c);
    for (auto& c : buildC3AttitudeSweep(200, 7)) cases.push_back(c);
    for (auto& c : buildC4IllConditioned()) cases.push_back(c);
    cases.push_back(buildC5UnequalScale());
    for (auto& c : buildC6ScalarCounterexamples()) cases.push_back(c);

    for (auto& c : cases) exportCaseCsv("outputs/cases", c);

    const std::vector<std::string> methods = {
        "SDA","SDA_FIXED","SDA_SS","ASDA","SDA_SCALED","SCHUR","VAN_DOOREN","ITERATIVE","ADDA",
        "SDA_SS_FIXED","ASDA_FIXED","SDA_SCALED_FIXED","ADDA_FIXED","ITERATIVE_FIXED"
    };
    // Todos os métodos fixed-point Q13.18 têm buffers dimensionados só p/ 6x3
    // (mesmo gate de computeGainMatrixSDA_Fixed(), ver docs/auditoria_solvers_riccati.md).
    auto isFixedPointMethod = [](const std::string& m) {
        return m.size() > 6 && m.compare(m.size() - 6, 6, "_FIXED") == 0;
    };

    std::ofstream out("outputs/verify_host_baseline.csv");
    out << "case,method,ok,iterations,lib_residual,dare_residual,symmetry_err,lambda_min_P,rho_closed_loop,time_us\n";

    int total = 0;
    for (const auto& c : cases) {
        for (const auto& method : methods) {
            if (isFixedPointMethod(method) && (c.n != 6 || c.m != 3)) continue;
            Metrics met = evaluate(c, method);
            out << met.case_name << "," << met.method << "," << (met.ok?1:0) << ","
                << met.iterations << "," << met.lib_residual << "," << met.dare_residual << ","
                << met.symmetry_err << "," << met.lambda_min_P << "," << met.rho_closed_loop << ","
                << met.time_us << "\n";
            total++;
        }
    }
    out.close();

    std::printf("OK: %d casos x metodo avaliados -> outputs/verify_host_baseline.csv\n", total);
    std::printf("Casos exportados em outputs/cases/ (%zu arquivos)\n", cases.size());
    return 0;
}
