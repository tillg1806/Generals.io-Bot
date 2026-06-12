import json
import sys
import time
from pathlib import Path

from sim_strategy_agent import SimStrategyAgent


SIM_ACTION_SAMPLES_FILE = Path("data/training/sim_action_samples.jsonl")


def ensure_generals_bots_path(sim_path=None):
    candidates = []
    if sim_path:
        candidates.append(Path(sim_path))
    env_path = _env_path("GENERALS_BOTS_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(Path.cwd().parent / "Generals-Bot-Sim" / "generals-bots")

    for candidate in candidates:
        if (candidate / "generals").exists():
            path_text = str(candidate)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
            return candidate

    return None


def run_sim_benchmark(
    games=32,
    parallel_games=32,
    grid_size=10,
    truncation=500,
    opponent="expander",
    sim_path=None,
    output_path=SIM_ACTION_SAMPLES_FILE,
):
    sim_root = ensure_generals_bots_path(sim_path)
    if sim_root is None:
        raise RuntimeError(
            "Could not find generals-bots. Set GENERALS_BOTS_PATH or pass --sim-path."
        )

    import jax.numpy as jnp
    import jax.random as jrandom
    from generals import GeneralsEnv, get_observation

    env = GeneralsEnv(
        grid_dims=(grid_size, grid_size),
        truncation=truncation,
        pool_size=max(128, parallel_games * 4),
        min_generals_distance=max(3, grid_size // 2),
    )
    key = jrandom.PRNGKey(int(time.time()) & 0xFFFFFFFF)
    pool, _ = env.reset(key)
    opponent_agent = make_sim_opponent(opponent)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    finished_games = 0
    total_steps = 0
    wins = 0
    losses = 0
    draws = 0
    pending_records = []

    while finished_games < games:
        batch_size = min(parallel_games, games - finished_games)
        init_keys = jrandom.split(key, batch_size + 1)
        key = init_keys[0]
        states = [env.init_state(init_keys[index + 1]) for index in range(batch_size)]
        agents = [
            SimStrategyAgent(player_index=0, label=f"sim-{finished_games + index + 1:05d}")
            for index in range(batch_size)
        ]
        done = [False] * batch_size
        last_infos = [None] * batch_size

        for _ in range(truncation):
            if all(done):
                break
            for index, state in enumerate(states):
                if done[index]:
                    continue

                obs0 = get_observation(state, 0)
                obs1 = get_observation(state, 1)
                key, k0, k1 = jrandom.split(key, 3)
                action0 = agents[index].act(obs0, k0)
                action1 = opponent_agent.act(obs1, k1)
                timestep, next_state = env.step(state, jnp.stack([action0, action1]), pool)
                states[index] = next_state
                last_infos[index] = timestep.info
                total_steps += 1

                if bool(timestep.terminated) or bool(timestep.truncated):
                    done[index] = True
                    winner = int(timestep.info.winner)
                    if winner == 0:
                        wins += 1
                    elif winner == 1:
                        losses += 1
                    else:
                        draws += 1
                    pending_records.extend(
                        agents[index].finished_records(
                            won=(winner == 0),
                            opponent=opponent,
                            final_info=timestep.info,
                        )
                    )

        for index, is_done in enumerate(done):
            if is_done:
                continue
            info = last_infos[index]
            winner = int(info.winner) if info is not None else -1
            if winner == 0:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
            pending_records.extend(
                agents[index].finished_records(
                    won=(winner == 0) if winner >= 0 else None,
                    opponent=opponent,
                    final_info=info,
                )
            )

        append_jsonl(output_path, pending_records)
        pending_records = []
        finished_games += batch_size

    elapsed = max(0.001, time.perf_counter() - start)
    return {
        "status": "finished",
        "sim_root": str(sim_root),
        "games": games,
        "parallel_games": parallel_games,
        "grid_size": grid_size,
        "truncation": truncation,
        "opponent": opponent,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / games, 4) if games else 0.0,
        "steps": total_steps,
        "elapsed_seconds": round(elapsed, 3),
        "steps_per_second": round(total_steps / elapsed, 1),
        "output_path": str(output_path),
    }


def make_sim_opponent(name):
    from generals.agents import ExpanderAgent, HunterAgent, RandomAgent

    if name == "random":
        return RandomAgent()
    if name == "hunter":
        return HunterAgent()
    if name == "expander":
        return ExpanderAgent()
    raise ValueError(f"Unknown simulator opponent: {name}")


def append_jsonl(path, records):
    if not records:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _env_path(name):
    import os

    value = os.environ.get(name)
    return Path(value) if value else None
