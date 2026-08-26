"""
Orquestrador da campanha de medicao (SDRE_VECTORIZED / artigo DINAME 2027).

Fluxo pretendido: conectar o ESP32-S2 no PC, rodar um comando, obter os
resultados prontos para analise (figuras + RESULTS.md), opcionalmente ate o
PDF do artigo recompilado. Ver experiments/README.md para o guia de uso.

Uso rapido:
    python python/run_experiments.py --list
    python python/run_experiments.py --all
    python python/run_experiments.py --only gamma --force --no-pdf
    python python/run_experiments.py --analyze-only
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
PYTHON_DIR = REPO / "python"
ARTIGO_DIR = Path(
    r"G:\Meu Drive\ACADEMICO\Mestrado\EVENTOS\DINAME_2027\artigo_diname"
)
ARTIGO_TEX = "diname2027_v3.tex"

sys.path.insert(0, str(PYTHON_DIR))


# ---------------------------------------------------------------------------
# Registro de experimentos
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    key: str
    env: str                     # env do PlatformIO (compilar + gravar)
    outfile: str                 # nome do arquivo em outputs/
    markers: List[str]           # linhas de conclusao emitidas pelo firmware; [] = janela de tempo
    baud: int
    est_minutes: int
    description: str
    analysis: Optional[Callable[["RunContext"], None]] = None
    requires_imu: bool = False


def _run_gamma(ctx: "RunContext") -> None:
    ctx.run_python(["python/analisa_gamma.py", "--device", str(OUTPUTS / "serial_gamma_sweep.txt")])


def _run_tolerancia(ctx: "RunContext") -> None:
    ctx.run_python(
        ["python/analisa_tolerancia.py", "--device", str(OUTPUTS / "serial_tolerance_sweep_frobenius.txt")]
    )


def _run_sweep_qr(ctx: "RunContext") -> None:
    ctx.run_python(["python/analisa_sweep_qr.py", str(OUTPUTS / "serial_sweep_qr_v4.txt")])


def _run_bateria(ctx: "RunContext") -> None:
    # A bateria alimenta o relatorio junto com a malha fechada (fase de host,
    # ver EXPERIMENTS_HOST abaixo) — aqui so garantimos que a captura existe;
    # a analise consolidada roda na fase final (_run_report).
    return


EXPERIMENTS: List[Experiment] = [
    Experiment(
        key="tolerancia",
        env="tolerance_sweep",
        outfile="serial_tolerance_sweep_frobenius.txt",
        markers=["FIM DA VARREDURA DE TOLERANCIA"],
        baud=921600,
        est_minutes=20,
        description="Exp. 0 - tolerancia/orcamento casados entre float e ponto fixo",
        analysis=_run_tolerancia,
    ),
    Experiment(
        key="gamma",
        env="gamma_sweep",
        outfile="serial_gamma_sweep.txt",
        markers=["FIM DA VARREDURA DE GAMMA"],
        baud=921600,
        est_minutes=5,
        description="Exp. 3 - varredura de gamma no SDA-SS",
        analysis=_run_gamma,
    ),
    Experiment(
        key="sweep_qr",
        env="sweep_qr",
        outfile="serial_sweep_qr_v4.txt",
        markers=["FIM DA VARREDURA"],
        baud=921600,
        est_minutes=27,
        description="Mapa de seguranca Q/R (faixa dinamica e overflow dos solvers _FIXED)",
        analysis=_run_sweep_qr,
    ),
    Experiment(
        key="tol_qr",
        env="tol_qr_sweep",
        outfile="serial_tol_qr_sweep_A.txt",
        markers=["FIM DA VARREDURA TAU x QR"],
        baud=921600,
        est_minutes=280,
        description="Exp. A - varredura combinada tau x Q/R (o carro-chefe)",
    ),
    Experiment(
        key="fronteiras",
        env="boundary_fine",
        outfile="serial_boundary_fine_B.txt",
        markers=["FIM DO MAPA FINO"],
        baud=921600,
        est_minutes=83,
        description="Exp. B - mapa fino das duas fronteiras de falha",
    ),
    Experiment(
        key="repetibilidade",
        env="repeatability",
        outfile="serial_repeatability_D.txt",
        markers=["FIM DA REPETIBILIDADE"],
        baud=921600,
        est_minutes=58,
        description="Exp. D - repetibilidade/jitter (20 repeticoes por ponto)",
    ),
    Experiment(
        key="bateria",
        env="benchmark",
        outfile="serial_capture_bateria_v5_6traj.txt",
        markers=["FIM DO BENCHMARK"],
        baud=921600,
        est_minutes=105,
        description="Exp. 1 - bateria principal (69228 pontos, 6 trajetorias x 12 metodos)",
        analysis=_run_bateria,
    ),
    Experiment(
        key="voo",
        env="esp32-s2-saola-1",
        outfile="serial_flightloop_E.txt",
        markers=[],  # janela de tempo — o firmware de voo nao emite marcador de fim
        baud=115200,
        est_minutes=6,
        description="Exp. E - ciclo de voo completo (IMU conectada, motores desarmados)",
        requires_imu=True,
    ),
]

EXPERIMENTS_BY_KEY = {e.key: e for e in EXPERIMENTS}


# ---------------------------------------------------------------------------
# Contexto de execucao: logging, subprocessos, estado
# ---------------------------------------------------------------------------

class RunContext:
    def __init__(self, args: argparse.Namespace, log_path: Optional[Path]):
        self.args = args
        self.log_path = log_path
        self._log_fh = open(log_path, "a", encoding="utf-8") if log_path else None
        self.status: dict[str, str] = {}
        self.timings: dict[str, float] = {}
        self.pio_exe: str = "pio"

    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self._log_fh:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()

    def close(self) -> None:
        if self._log_fh:
            self._log_fh.close()

    def run(self, cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
        self.log("$ " + " ".join(cmd))
        if self.args.dry_run:
            return subprocess.CompletedProcess(cmd, 0)
        proc = subprocess.run(cmd, cwd=str(cwd or REPO), text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(f"comando falhou (exit {proc.returncode}): {' '.join(cmd)}")
        return proc

    def run_python(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        return self.run([sys.executable] + args, check=check)


# ---------------------------------------------------------------------------
# Pre-checagens
# ---------------------------------------------------------------------------

def find_pio_exe() -> str:
    """Localiza o executavel real do PlatformIO Core.

    Em alguns setups Windows existe um pacote pip chamado 'pio' (sem relacao
    com o PlatformIO) que sombra o comando no PATH e falha com
    'ModuleNotFoundError: No module named requirements'. Preferimos o venv
    dedicado do PlatformIO (~/.platformio/penv) quando ele existe, e so caimos
    para o PATH genérico se ele nao estiver presente.
    """
    home = Path.home()
    preferred = (
        home / ".platformio" / "penv" / "Scripts" / "platformio.exe"
        if os.name == "nt"
        else home / ".platformio" / "penv" / "bin" / "platformio"
    )
    if preferred.exists():
        return str(preferred)

    found = shutil.which("platformio") or shutil.which("pio")
    if not found:
        raise SystemExit(
            "Executavel do PlatformIO nao encontrado (nem em ~/.platformio/penv nem no "
            "PATH). Instale o PlatformIO Core: https://platformio.org/install/cli"
        )
    return found


def preflight(ctx: RunContext, need_hardware: bool) -> str:
    ctx.log("Pre-checagem: dependencias Python...")
    missing = []
    for mod in ("serial", "numpy", "scipy", "matplotlib"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"Dependencias Python ausentes: {', '.join(missing)}. "
            "Instale com: pip install -r requirements.txt"
        )

    port = ""
    if need_hardware:
        ctx.log("Pre-checagem: executavel do PlatformIO...")
        ctx.pio_exe = find_pio_exe()
        ctx.log(f"  usando: {ctx.pio_exe}")

        ctx.log("Pre-checagem: porta serial...")
        from serial_capture import find_esp32_port  # type: ignore

        port = find_esp32_port(ctx.args.port)
        ctx.log(f"  porta detectada: {port}")

        ctx.log("Pre-checagem: espaco em disco...")
        usage = shutil.disk_usage(OUTPUTS if OUTPUTS.exists() else REPO)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < 500:
            raise SystemExit(
                f"Espaco em disco insuficiente ({free_mb:.0f} MB livres); a campanha "
                "completa gera ~150 MB de captura. Libere espaco antes de continuar."
            )
        ctx.log(f"  espaco livre: {free_mb:.0f} MB")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ctx.log("Pre-checagem OK.")
    return port


# ---------------------------------------------------------------------------
# Execucao de um experimento de placa
# ---------------------------------------------------------------------------

def run_board_experiment(ctx: RunContext, exp: Experiment, port: str) -> None:
    from serial_capture import capture  # type: ignore

    outfile = OUTPUTS / exp.outfile
    if outfile.exists() and not ctx.args.force:
        ctx.log(f"[{exp.key}] captura ja existe ({outfile.name}), pulando (use --force para refazer).")
        ctx.status[exp.key] = "pulado (ja existia)"
        return

    if exp.requires_imu:
        ctx.log(
            f"[{exp.key}] ATENCAO: este experimento precisa da IMU conectada - sem ela o "
            "firmware trava em while(1) (lib/utils/utils.cpp:19-24). Motores ficam desarmados "
            "durante toda a medicao."
        )

    ctx.log(f"[{exp.key}] compilando env '{exp.env}'...")
    ctx.run([ctx.pio_exe, "run", "-e", exp.env])

    ctx.log(f"[{exp.key}] gravando na placa ({port})...")
    ctx.run([ctx.pio_exe, "run", "-e", exp.env, "-t", "upload", "--upload-port", port])

    ctx.log(
        f"[{exp.key}] capturando a {exp.baud} baud (estimativa ~{exp.est_minutes} min)"
        + (f", ate o marcador {exp.markers!r}" if exp.markers else ", por janela de tempo fixa")
        + "..."
    )
    if ctx.args.dry_run:
        ctx.status[exp.key] = "dry-run"
        return

    t0 = time.time()
    timeout_s = max(exp.est_minutes * 60 * 3, 300)  # teto generoso: 3x a estimativa
    result = capture(
        port,
        exp.baud,
        str(outfile),
        markers=exp.markers or None,
        timeout_s=timeout_s,
        progress_cb=lambda line: ctx.log(f"  {line}"),
    )
    elapsed = time.time() - t0
    ctx.timings[exp.key] = elapsed

    if exp.markers and not result.done_flag:
        ctx.status[exp.key] = f"FALHOU (timeout sem marcador, {elapsed:.0f}s)"
        ctx.log(f"[{exp.key}] captura NAO viu o marcador de conclusao em {timeout_s}s - verifique o log.")
        return

    ctx.status[exp.key] = f"OK ({elapsed:.0f}s, {result.n_bytes} bytes)"
    ctx.log(f"[{exp.key}] captura concluida: {result.n_bytes} bytes em {elapsed:.0f}s.")

    if exp.analysis:
        ctx.log(f"[{exp.key}] rodando analise dedicada...")
        try:
            exp.analysis(ctx)
        except Exception as e:  # nao aborta a campanha por causa de uma analise
            ctx.log(f"[{exp.key}] analise dedicada falhou: {e}")


# ---------------------------------------------------------------------------
# Exp. voo — caso especial: alterna DEBUG_MODE, sempre restaura
# ---------------------------------------------------------------------------

MAIN_CPP = REPO / "src" / "main.cpp"


def _read_debug_mode() -> bool:
    text = MAIN_CPP.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "const bool DEBUG_MODE" in line:
            return "true" in line.split("=", 1)[1].split(";", 1)[0]
    raise RuntimeError("Nao encontrei 'const bool DEBUG_MODE' em src/main.cpp")


def _set_debug_mode(value: bool) -> None:
    text = MAIN_CPP.read_text(encoding="utf-8")
    target = "true" if value else "false"
    new_text, n = re.subn(
        r"(const bool DEBUG_MODE\s*=\s*)(true|false)(\s*;)",
        rf"\g<1>{target}\g<3>",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Nao encontrei 'const bool DEBUG_MODE = true|false;' em src/main.cpp")
    MAIN_CPP.write_text(new_text, encoding="utf-8")


def _build_and_upload_flight(ctx: RunContext, port: str) -> None:
    ctx.run([ctx.pio_exe, "run", "-e", "esp32-s2-saola-1", "-t", "upload", "--upload-port", port])


@contextlib.contextmanager
def flight_debug_mode(ctx: RunContext, port: str):
    """Liga DEBUG_MODE, recompila e regrava a placa; ao sair (mesmo com
    excecao) restaura o valor original, recompila e REGRAVA de volta —
    o firmware de voo nunca fica fisicamente em estado de depuracao por
    causa de uma falha no meio do caminho (a placa e' o estado que importa,
    nao so o arquivo fonte)."""
    original = _read_debug_mode()
    ctx.log(f"[voo] DEBUG_MODE original = {original}")
    dry = ctx.args.dry_run
    try:
        if not original:
            ctx.log(f"[voo] ligando DEBUG_MODE=true, recompilando e gravando na placa ({port})...")
            if not dry:
                _set_debug_mode(True)
            _build_and_upload_flight(ctx, port)
        yield
    finally:
        if not original:
            ctx.log(f"[voo] restaurando DEBUG_MODE=false, recompilando e REGRAVANDO na placa ({port})...")
            if not dry:
                _set_debug_mode(False)
            try:
                _build_and_upload_flight(ctx, port)
                ctx.log("[voo] placa restaurada para DEBUG_MODE=false.")
            except Exception as e:
                ctx.log(
                    f"[voo] AVISO CRITICO: restauracao (recompilar+regravar) falhou: {e}. "
                    "src/main.cpp ja esta com DEBUG_MODE=false, mas a PLACA pode ainda estar "
                    "rodando o firmware de depuracao. Rode manualmente: "
                    f"pio run -e esp32-s2-saola-1 -t upload --upload-port {port}"
                )


def run_flight_experiment(ctx: RunContext, exp: Experiment, port: str) -> None:
    outfile = OUTPUTS / exp.outfile
    if outfile.exists() and not ctx.args.force:
        ctx.log(f"[{exp.key}] captura ja existe ({outfile.name}), pulando (use --force para refazer).")
        ctx.status[exp.key] = "pulado (ja existia)"
        return

    ctx.log(
        "[voo] IMPORTANTE: IMU precisa estar conectada (senao o firmware trava em "
        "while(1), lib/utils/utils.cpp:19-24) e os motores permanecem desarmados "
        "durante toda a captura."
    )

    with flight_debug_mode(ctx, port):
        from serial_capture import capture  # type: ignore

        window_s = max(exp.est_minutes * 60, 120)
        ctx.log(f"[voo] capturando por janela fixa de {window_s}s a {exp.baud} baud...")
        if ctx.args.dry_run:
            ctx.status[exp.key] = "dry-run"
            return

        t0 = time.time()
        result = capture(port, exp.baud, str(outfile), markers=None, timeout_s=window_s,
                          progress_cb=lambda line: ctx.log(f"  {line}"))
        elapsed = time.time() - t0
        ctx.timings[exp.key] = elapsed
        ctx.status[exp.key] = f"OK ({elapsed:.0f}s, {result.n_bytes} bytes)"
        ctx.log(f"[voo] captura concluida: {result.n_bytes} bytes em {elapsed:.0f}s.")


# ---------------------------------------------------------------------------
# Fases de host (nao usam hardware)
# ---------------------------------------------------------------------------

def run_host_phases(ctx: RunContext) -> None:
    if ctx.args.dry_run:
        ctx.log("(--dry-run) fases de host: malha fechada, cobertura, relatorio da bateria.")
        for k in ("malha_fechada", "cobertura", "relatorio_bateria"):
            ctx.status[k] = "dry-run"
        return

    ctx.log("Fase de host: malha fechada (trajetorias x float64 vs Q13.18)...")
    try:
        ctx.run_python([
            "python/malha_fechada_trajetorias.py",
            "--saida", str(OUTPUTS / "malha_fechada_v6_6traj.csv"),
        ])
        ctx.status["malha_fechada"] = "OK"
    except Exception as e:
        ctx.status["malha_fechada"] = f"FALHOU: {e}"
        ctx.log(f"malha fechada falhou: {e}")

    ctx.log("Fase de host: cobertura (condicionamento das 6 trajetorias)...")
    try:
        ctx.run_python([
            "python/analisa_cobertura.py",
            "--saida", str(OUTPUTS / "cobertura_full_v5_6traj.csv"),
        ])
        ctx.status["cobertura"] = "OK"
    except Exception as e:
        ctx.status["cobertura"] = f"FALHOU: {e}"
        ctx.log(f"cobertura falhou: {e}")

    ctx.log("Fase de host: relatorio da bateria (tempos + erro de K + malha fechada)...")
    try:
        ctx.run_python(["python/gerar_relatorio_bateria.py"])
        ctx.status["relatorio_bateria"] = "OK"
    except Exception as e:
        ctx.status["relatorio_bateria"] = f"FALHOU: {e}"
        ctx.log(f"relatorio da bateria falhou: {e}")


def run_figures(ctx: RunContext) -> None:
    if ctx.args.dry_run:
        ctx.log("(--dry-run) geracao de figuras.")
        ctx.status["figuras"] = "dry-run"
        return

    ctx.log("Gerando figuras do artigo (figuras_artigo_final.py)...")
    try:
        ctx.run_python(["python/figuras_artigo_final.py"])
        ctx.status["figuras"] = "OK"
    except Exception as e:
        ctx.status["figuras"] = f"FALHOU: {e}"
        ctx.log(f"geracao de figuras falhou: {e}")


def run_article_pdf(ctx: RunContext) -> None:
    if ctx.args.dry_run:
        ctx.log("(--dry-run) recompilacao do PDF do artigo.")
        ctx.status["pdf"] = "dry-run"
        return

    if not ARTIGO_DIR.exists():
        ctx.log(f"AVISO: diretorio do artigo nao encontrado ({ARTIGO_DIR}); pulando compilacao do PDF.")
        ctx.status["pdf"] = "pulado (diretorio nao encontrado)"
        return

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex or not bibtex:
        ctx.log("AVISO: pdflatex/bibtex nao encontrados no PATH (instale o MiKTeX). Pulando compilacao do PDF.")
        ctx.status["pdf"] = "pulado (MiKTeX ausente)"
        return

    stem = ARTIGO_TEX[:-4]
    ctx.log(f"Recompilando o artigo ({ARTIGO_TEX})...")
    try:
        ctx.run(["pdflatex", "-interaction=nonstopmode", ARTIGO_TEX], cwd=ARTIGO_DIR, check=False)
        ctx.run(["bibtex", stem], cwd=ARTIGO_DIR, check=False)
        ctx.run(["pdflatex", "-interaction=nonstopmode", ARTIGO_TEX], cwd=ARTIGO_DIR, check=False)
        ctx.run(["pdflatex", "-interaction=nonstopmode", ARTIGO_TEX], cwd=ARTIGO_DIR, check=False)
        pdf_path = ARTIGO_DIR / f"{stem}.pdf"
        if pdf_path.exists():
            ctx.status["pdf"] = f"OK ({pdf_path})"
            ctx.log(f"PDF recompilado: {pdf_path}")
        else:
            ctx.status["pdf"] = "FALHOU (PDF nao foi gerado - ver .log no diretorio do artigo)"
            ctx.log("AVISO: pdflatex nao produziu o PDF. Verifique o .log no diretorio do artigo.")
    except Exception as e:
        ctx.status["pdf"] = f"FALHOU: {e}"
        ctx.log(f"compilacao do PDF falhou: {e}")


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------

def write_results_md(ctx: RunContext) -> None:
    lines = [
        "# RESULTS.md - campanha de medicao SDRE_VECTORIZED",
        "",
        f"Gerado em {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} por python/run_experiments.py.",
        "",
        "## Status por experimento",
        "",
        "| experimento | status | duracao |",
        "|---|---|---|",
    ]
    for exp in EXPERIMENTS:
        status = ctx.status.get(exp.key, "nao executado")
        dur = ctx.timings.get(exp.key)
        dur_s = f"{dur:.0f}s" if dur is not None else "-"
        lines.append(f"| {exp.key} | {status} | {dur_s} |")

    lines.append("")
    lines.append("## Fases de host e artigo")
    lines.append("")
    lines.append("| fase | status |")
    lines.append("|---|---|")
    for k in ("malha_fechada", "cobertura", "relatorio_bateria", "figuras", "pdf"):
        if k in ctx.status:
            lines.append(f"| {k} | {ctx.status[k]} |")

    lines.append("")
    lines.append("## Arquivos gerados em outputs/")
    lines.append("")
    for exp in EXPERIMENTS:
        p = OUTPUTS / exp.outfile
        if p.exists():
            lines.append(f"- `{exp.outfile}` ({p.stat().st_size:,} bytes)")

    out_path = OUTPUTS / "RESULTS.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx.log(f"RESULTS.md escrito em {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_list() -> None:
    print(f"{'chave':16s} {'env':16s} {'~min':>5s}  descricao")
    print("-" * 90)
    for exp in EXPERIMENTS:
        outfile = OUTPUTS / exp.outfile
        status = "capturado" if outfile.exists() else "pendente"
        print(f"{exp.key:16s} {exp.env:16s} {exp.est_minutes:5d}  [{status:9s}] {exp.description}")
    total = sum(e.est_minutes for e in EXPERIMENTS)
    print("-" * 90)
    print(f"tempo total estimado (campanha completa): ~{total} min (~{total/60:.1f} h)")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--list", action="store_true", help="lista os experimentos e sai")
    ap.add_argument("--all", action="store_true", help="roda a campanha completa (~10 h)")
    ap.add_argument("--only", nargs="+", metavar="KEY", help="roda so estes experimentos")
    ap.add_argument("--skip", nargs="+", metavar="KEY", default=[], help="pula estes experimentos")
    ap.add_argument("--force", action="store_true", help="recaptura mesmo se o arquivo ja existir")
    ap.add_argument("--port", default=None, help="porta serial (padrao: autodetecao)")
    ap.add_argument("--analyze-only", action="store_true",
                     help="pula toda a medicao; roda so analise + figuras + PDF sobre capturas existentes")
    ap.add_argument("--no-pdf", action="store_true", help="para antes de recompilar o artigo")
    ap.add_argument("--dry-run", action="store_true", help="mostra o que faria, sem tocar em nada")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.list:
        print_list()
        return

    if not (args.all or args.only or args.analyze_only):
        ap.print_help()
        raise SystemExit(2)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUTS / f"run_{timestamp}.log"
    ctx = RunContext(args, log_path if not args.dry_run else None)

    try:
        need_hardware = not args.analyze_only
        port = preflight(ctx, need_hardware=need_hardware)

        if not args.analyze_only:
            if args.only:
                keys = args.only
                unknown = [k for k in keys if k not in EXPERIMENTS_BY_KEY]
                if unknown:
                    raise SystemExit(f"chave(s) desconhecida(s): {unknown}. Use --list para ver as validas.")
                selected = [EXPERIMENTS_BY_KEY[k] for k in keys]
            else:
                selected = list(EXPERIMENTS)

            selected = [e for e in selected if e.key not in args.skip]

            ctx.log(f"Experimentos selecionados: {[e.key for e in selected]}")

            for exp in selected:
                ctx.log(f"=== {exp.key}: {exp.description} ===")
                try:
                    if exp.key == "voo":
                        run_flight_experiment(ctx, exp, port)
                    else:
                        run_board_experiment(ctx, exp, port)
                except Exception as e:
                    ctx.status[exp.key] = f"FALHOU: {e}"
                    ctx.log(f"[{exp.key}] ERRO: {e}")

        run_full_pipeline = args.all or args.analyze_only
        if run_full_pipeline:
            run_host_phases(ctx)
            run_figures(ctx)
            if not args.no_pdf:
                run_article_pdf(ctx)
            else:
                ctx.status["pdf"] = "pulado (--no-pdf)"
        else:
            ctx.log(
                "--only: pulando fases de host, figuras e PDF (use --all para o pipeline "
                "completo). A analise dedicada de cada experimento, se houver, ja rodou acima."
            )

        if args.dry_run:
            ctx.log("(--dry-run: RESULTS.md nao foi escrito.)")
        else:
            write_results_md(ctx)
        ctx.log("Campanha finalizada.")

    finally:
        ctx.close()


if __name__ == "__main__":
    main()
