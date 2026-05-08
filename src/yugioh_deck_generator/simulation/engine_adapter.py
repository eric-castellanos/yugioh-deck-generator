from __future__ import annotations

import hashlib
import random
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from yugioh_deck_generator.simulation.schemas import DuelResult, SimulationRequest


@dataclass(slots=True)
class EngineAdapter:
    engine_mode: str = "stub"
    engine_version: str = "stub-v1"
    windbot_exe_path: str | None = None
    windbot_cards_cdb_path: str | None = None
    windbot_mono_path: str = "mono"
    windbot_host: str = "127.0.0.1"
    windbot_port: int = 7911
    windbot_host_info: str = ""
    windbot_dialog: str = "default"
    windbot_chat: bool = False
    windbot_debug: bool = False
    windbot_timeout_sec: int = 30
    ygoenv_task_id: str = "YGOPro-v1"
    ygoenv_db_path: str | None = None
    ygoenv_code_list_file: str | None = None
    ygoenv_max_steps: int = 2000

    def validate_environment(self) -> None:
        if self.engine_mode == "stub":
            return
        if self.engine_mode == "ygoenv":
            if not self.ygoenv_db_path:
                raise ValueError("--ygoenv-db-path is required for --engine-mode ygoenv")
            db = Path(self.ygoenv_db_path)
            if not db.exists():
                raise FileNotFoundError(f"ygoenv cards db not found: {db}")
            if self.ygoenv_code_list_file:
                code_list = Path(self.ygoenv_code_list_file)
                if not code_list.exists():
                    raise FileNotFoundError(f"ygoenv code list not found: {code_list}")
            try:
                import ygoenv  # noqa: F401
                from ygoenv.ygopro import init_module  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import ygoenv. Install ygoenv and provide ygopro_ygoenv.so."
                ) from exc
            return
        if self.engine_mode != "windbot":
            raise NotImplementedError("Supported engine modes are: stub, windbot, ygoenv.")
        if not self.windbot_exe_path:
            raise ValueError("--windbot-exe-path is required for --engine-mode windbot")
        exe = Path(self.windbot_exe_path)
        if not exe.exists():
            raise FileNotFoundError(f"WindBot executable not found: {exe}")
        if not shutil.which(self.windbot_mono_path):
            raise RuntimeError(f"Mono executable not found in PATH: {self.windbot_mono_path}")

        if self.windbot_cards_cdb_path:
            src_cdb = Path(self.windbot_cards_cdb_path)
            if not src_cdb.exists():
                raise FileNotFoundError(f"cards.cdb not found: {src_cdb}")
            dst_cdb = exe.parent / "cards.cdb"
            if not dst_cdb.exists():
                shutil.copy2(src_cdb, dst_cdb)

    def _run_stub(self, request: SimulationRequest) -> DuelResult:
        # Deterministic pseudo-outcome placeholder.
        payload = (
            f"{request.deck_id}|{request.opponent_deck_id}|{request.seed}|"
            f"{request.generation_run_id}|{request.scenario}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16)
        winner = "self" if (value % 2) == 0 else "opponent"
        return DuelResult(
            simulation_id=request.simulation_id,
            deck_id=request.deck_id,
            opponent_deck_id=request.opponent_deck_id,
            winner=winner,
            status="completed",
            seed=request.seed,
            engine_version=self.engine_version,
            error="",
        )

    def _run_windbot(self, request: SimulationRequest) -> DuelResult:
        exe = Path(self.windbot_exe_path or "")
        runtime_dir = exe.parent
        deck_dir = runtime_dir / "Decks"
        deck_dir.mkdir(parents=True, exist_ok=True)

        self_deck_name = f"SIM_SELF_{request.simulation_id}"
        staged_self_deck = deck_dir / f"{self_deck_name}.ydk"
        shutil.copy2(request.deck_path, staged_self_deck)

        # Also stage opponent deck for host-side orchestration compatibility.
        staged_opp_deck = deck_dir / f"SIM_OPP_{request.simulation_id}.ydk"
        shutil.copy2(request.opponent_deck_path, staged_opp_deck)

        cmd = [
            self.windbot_mono_path,
            str(exe),
            f"Name=SIM_{request.deck_id[:8]}",
            f"Deck={self_deck_name}",
            f"Host={self.windbot_host}",
            f"Port={int(self.windbot_port)}",
            f"HostInfo={self.windbot_host_info}",
            f"Dialog={self.windbot_dialog}",
            f"Chat={str(self.windbot_chat).lower()}",
            f"Debug={str(self.windbot_debug).lower()}",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(runtime_dir),
                capture_output=True,
                text=True,
                timeout=max(1, int(self.windbot_timeout_sec)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return DuelResult(
                simulation_id=request.simulation_id,
                deck_id=request.deck_id,
                opponent_deck_id=request.opponent_deck_id,
                winner="timeout",
                status="failed",
                seed=request.seed,
                engine_version=self.engine_version,
                error="windbot subprocess timed out",
            )

        output_text = f"{proc.stdout}\n{proc.stderr}".lower()
        # Best-effort parsing; depends on host/log format.
        if "win" in output_text and "lose" not in output_text:
            winner = "self"
            status = "completed"
            err = ""
        elif "lose" in output_text or "lost" in output_text:
            winner = "opponent"
            status = "completed"
            err = ""
        elif proc.returncode == 0:
            winner = "invalid"
            status = "failed"
            err = "windbot ran but no explicit duel outcome parsed"
        else:
            winner = "invalid"
            status = "failed"
            err = f"windbot exit_code={proc.returncode}"

        return DuelResult(
            simulation_id=request.simulation_id,
            deck_id=request.deck_id,
            opponent_deck_id=request.opponent_deck_id,
            winner=winner,
            status=status,
            seed=request.seed,
            engine_version=self.engine_version,
            error=err,
        )

    def _run_ygoenv(self, request: SimulationRequest) -> DuelResult:
        import numpy as np
        import ygoenv
        from ygoenv.ygopro import init_module

        deck1_name = f"SELF_{request.simulation_id[:12]}"
        deck2_name = f"OPP_{request.simulation_id[:12]}"
        decks = {
            deck1_name: str(request.deck_path),
            deck2_name: str(request.opponent_deck_path),
        }
        code_list_file = self._resolve_ygoenv_code_list_file()
        try:
            init_module(
                str(self.ygoenv_db_path),
                str(code_list_file),
                decks,
            )
        except Exception as exc:
            return DuelResult(
                simulation_id=request.simulation_id,
                deck_id=request.deck_id,
                opponent_deck_id=request.opponent_deck_id,
                winner="invalid",
                status="failed",
                seed=request.seed,
                engine_version=self.engine_version,
                error=f"ygoenv init_module failed: {exc}",
            )
        try:
            env = ygoenv.make(
                task_id=self.ygoenv_task_id,
                env_type="gymnasium",
                num_envs=1,
                num_threads=1,
                seed=int(request.seed),
                deck1=deck1_name,
                deck2=deck2_name,
                player=-1,
                max_options=24,
                n_history_actions=32,
                play_mode="random",
                async_reset=False,
                verbose=False,
                record=False,
            )
        except Exception as exc:
            return DuelResult(
                simulation_id=request.simulation_id,
                deck_id=request.deck_id,
                opponent_deck_id=request.opponent_deck_id,
                winner="invalid",
                status="failed",
                seed=request.seed,
                engine_version=self.engine_version,
                error=f"ygoenv make failed: {exc}",
            )
        _, infos = env.reset()
        rng = random.Random(int(request.seed))
        reward = 0.0
        win_reason = 0
        completed = False
        for _ in range(max(1, int(self.ygoenv_max_steps))):
            num_options = max(1, int(infos["num_options"][0]))
            action = np.asarray([rng.randint(0, num_options - 1)], dtype=np.int32)
            _, rewards, terminated, truncated, infos = env.step(action)
            reward = float(rewards[0])
            win_reason = int(infos["win_reason"][0])
            if bool(terminated[0] or truncated[0]):
                completed = True
                break
        if not completed:
            return DuelResult(
                simulation_id=request.simulation_id,
                deck_id=request.deck_id,
                opponent_deck_id=request.opponent_deck_id,
                winner="timeout",
                status="failed",
                seed=request.seed,
                engine_version=self.engine_version,
                error=f"ygoenv duel exceeded max steps ({self.ygoenv_max_steps})",
            )
        winner = "self" if reward > 0 else "opponent"
        return DuelResult(
            simulation_id=request.simulation_id,
            deck_id=request.deck_id,
            opponent_deck_id=request.opponent_deck_id,
            winner=winner,
            status="completed",
            seed=request.seed,
            engine_version=self.engine_version,
            error=f"win_reason={win_reason}",
        )

    def _resolve_ygoenv_code_list_file(self) -> str:
        if self.ygoenv_code_list_file:
            return str(self.ygoenv_code_list_file)
        db_path = Path(self.ygoenv_db_path or "")
        if not db_path.exists():
            raise FileNotFoundError(f"ygoenv cards db not found: {db_path}")

        out_path = Path(tempfile.gettempdir()) / "ygoenv_auto_code_list.txt"
        if out_path.exists():
            return str(out_path)

        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute("SELECT id FROM datas ORDER BY id ASC").fetchall()
        finally:
            conn.close()
        ids = [str(int(row[0])) for row in rows if row and int(row[0]) > 0]
        out_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        return str(out_path)

    def run_duel(self, request: SimulationRequest) -> DuelResult:
        if self.engine_mode == "stub":
            return self._run_stub(request)
        if self.engine_mode == "windbot":
            return self._run_windbot(request)
        if self.engine_mode == "ygoenv":
            return self._run_ygoenv(request)
        raise NotImplementedError(f"Unsupported engine mode: {self.engine_mode}")
