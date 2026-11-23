"""Run the three matched semantic supervision conditions on frozen Qwen states."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.core_features.fit import write_rows
from research.stage3.run import save_json
from .joint_inputs import load
from .joint_model import TASKS, SemanticHeads, objective, projection, parallel_trace


MODES = ('independent', 'shared_two', 'shared_three')
SEEDS = (43, 71, 109)
STEPS = 180
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.015
BATCH = {'onset': 96, 'source': 48, 'closure': 20}


def fit(tasks, training, seed, device):
    torch.manual_seed(seed)
    model = SemanticHeads(tasks).to(device)
    batches = {task: np.random.default_rng(seed + 1009 * (i + 1)).integers(
        0, len(training[task].keys), size=(STEPS, BATCH[task]),
    ) for i, task in enumerate(TASKS) if task in tasks}
    batches = {key: torch.as_tensor(value, device=device) for key, value in batches.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elapsed = perf_counter()
    selected = (0, STEPS // 2, STEPS - 1)
    diagnostics = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for task in tasks:
            x = training[task]
            indices = batches[task][step]
            logits = model(task, x.inputs[indices])
            losses[task] = objective(task, logits, x.mask[indices], x.labels[indices])
        sum(losses.values()).backward()
        optimizer.step()
        if step in selected:
            means = {task: value.item() for task, value in losses.items()}
            if not np.isfinite(list(means.values())).all():
                raise ArithmeticError(f'Nonfinite training loss for {tasks}')
            diagnostics.append({'step': step + 1, 'loss': means})
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return model.eval(), {'steps': STEPS, 'seed': seed, 'tasks': list(tasks), 'train_log': diagnostics,
                          'seconds': perf_counter() - elapsed}


def forecast(model, task, dataset):
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(dataset.keys), 80):
            x = dataset.inputs[start:start + 80]
            mask = dataset.mask[start:start + 80]
            logits = model(task, x).double()
            if task == 'source':
                probabilities.extend(torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1).cpu().numpy())
            else:
                probabilities.extend(torch.sigmoid(logits).cpu().numpy())
    scores = np.asarray(probabilities)
    if not np.isfinite(scores).all():
        raise ArithmeticError(f'Nonfinite {task} prediction')
    return scores


def serialize(task, data, predictions):
    lengths = data.mask.sum(dim=-1).tolist()
    for key, length, *values in zip(data.keys, lengths, *[predictions[mode] for mode in MODES], strict=True):
        if task == 'onset':
            payload = {mode: float(value[0]) for mode, value in zip(MODES, values, strict=True)}
        else:
            payload = {mode: value[:length].tolist() for mode, value in zip(MODES, values, strict=True)}
        yield {**key, **payload}


def fixture(model, device):
    rng = np.random.default_rng(5)
    common = rng.normal(size=(8, 2048)).astype(np.float32)
    changed = common.copy()
    changed[5:] += rng.normal(size=(3, 2048)).astype(np.float32)
    counts = []
    for states in (common, changed):
        def stub():
            counts.append(1)
            return states

        out = parallel_trace(model, stub, np.arange(1, 9), (1, 3), (1, 3, 5, 7), device)
        if len(counts) == 1:
            original = out
    error = float(np.max(np.abs(original['onset'][:5] - out['onset'][:5])))
    link_error = max(float(np.max(np.abs(a['source_probability'] - b['source_probability'])))
                     for a, b in zip(original['tracks'], out['tracks'], strict=True))
    closure_error = max(float(np.max(np.abs(a['closure_hazard'] - b['closure_hazard'])))
                        for a, b in zip(original['tracks'], out['tracks'], strict=True))
    if len(counts) != 2 or max(error, link_error, closure_error) > 1e-7:
        raise ValueError('One-call trace failed the shared-prefix fixture')
    return {'forward_calls_per_trace': 1, 'backbone_stub': True, 'opening_prefix_max_difference': error,
            'earlier_source_max_difference': link_error, 'position_only_closure_max_difference': closure_error,
            'anchors': [int(track['anchor_token']) for track in original['tracks']]}


def run(stimuli, features, input_dir, output):
    started = perf_counter()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Expected one NVIDIA L4')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')
    output.mkdir(parents=True, exist_ok=False)
    train, text_test, closure = load(stimuli, features, input_dir, device)
    text_forecasts = {task: {mode: [] for mode in MODES} for task in ('onset', 'source')}
    close_forecasts = []
    archive = {'projection': projection()}
    performance = []
    torch.cuda.reset_peak_memory_stats()
    for (story, (closure_train, closure_test)), seed in zip(closure.items(), SEEDS, strict=True):
        all_training = {**train, 'closure': closure_train}
        close_models = {}
        for mode in MODES:
            configurations = {
                'independent': [(task,) for task in TASKS], 'shared_two': [('onset', 'source')],
                'shared_three': [TASKS],
            }
            for tasks in configurations[mode]:
                model, evidence = fit(tasks, all_training, seed, device)
                key = '.'.join((story, mode, '+'.join(tasks)))
                for name, tensor in model.state_dict().items():
                    archive[f'{key}.{name}'] = tensor.detach().cpu().numpy()
                performance.append({'condition': mode, 'held_out_story': story, **evidence})
                for task in tasks:
                    if task == 'closure':
                        close_models[mode] = forecast(model, task, closure_test)
                    else:
                        text_forecasts[task][mode].append(forecast(model, task, text_test[task]))
                print(f'Fitted {story} {mode} {tasks}', flush=True)
        close_models['shared_two'] = close_models['independent']
        close_forecasts.extend(serialize('closure', closure_test, close_models))
    for task, splits in text_forecasts.items():
        mean = {name: np.mean(predictions, axis=0) for name, predictions in splits.items()}
        write_rows(output / f'{task}-predictions.jsonl', serialize(task, text_test[task], mean))
    write_rows(output / 'closure-predictions.jsonl', close_forecasts)
    save_json(output / 'trace-fixture.json', fixture(model, device))
    np.savez(output / 'joint-heads.npz', **archive)
    saves = [f'{task}-predictions.jsonl' for task in TASKS]
    save_json(output / 'joint-manifest.json', {
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'modes': list(MODES),
        'story_folds': list(closure), 'seeds': list(SEEDS), 'steps_per_fit': STEPS,
        'fit_records': performance, 'fit_wall_s': perf_counter() - started,
        'batch_sizes': BATCH, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
        'forecast_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in saves},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('stimuli', 'features', 'input-dir', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    run(**vars(parser.parse_args()))
