"""Summarize this campaign's verified single-accelerator allocations without account identifiers."""
from datetime import datetime, timezone
from pathlib import Path
import json

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.identity_colab import DATA, ROOT

DESTINATION = ROOT / 'research/results/reading-validation-identity/compute-ledger.json'


def ledger():
    rows = []
    pending = []
    no_session = []
    study_receipts = []
    specialized_archives = []
    interrupted = []
    manual = read_json(ROOT / 'research/results/reading-validation-identity-transfer/compute.json')['first_attempt_manual_cleanup']
    manually_closed = (manual['server_stop_exit_code'] == 0 and manual['subsequent_active_assignments'] == 0
                       and manual['subsequent_rate_cu_per_hour'] == 0)
    for path in sorted((DATA/'l4').glob('*/manifest.json')):
        record=read_json(path);times=record['timeline'];name=path.parent.name
        if 'allocated' not in times:
            if 'allocating' in times:no_session.append({'job':name,'phase':record['phase'],'released':record.get('release_verified')})
            continue
        if 'releasing' not in times:
            pending.append({'job':name,'start':times['allocated'],'phase':record['phase']})
            continue
        a=datetime.fromisoformat(times['allocated']);b=datetime.fromisoformat(times['releasing'])
        sec=(b-a).total_seconds();rates=[]
        for line in (path.parent/'usage.jsonl').open():
            q=json.loads(line)
            # Use only the observed reported rate when the sole billed accelerator was active.
            if q['account_active_assignments']==1 and q['account_rate_cu_per_hour']>0:
                rates.append(float(q['account_rate_cu_per_hour']))
        if len(set(rates))>1:raise ValueError('Billing rate changed within an allocation')
        verified=(record.get('release_verified') is True or name=='cogllm-reading-transfer-20261122T154850Z-6587fd' and manually_closed)
        receipt = path.parent / 'extraction/causality.json'
        if receipt.is_file():
            phases = read_json(receipt).get('studies')
            if phases:
                for phase, status in phases.items():
                    if not isinstance(status.get('completed'), bool):
                        raise ValueError(f'Invalid study completion field for {phase}')
                    study_receipts.append({'job': name, 'phase': phase, 'completed': status['completed']})
            else:
                specialized_archives.append(name)
        else:
            interrupted.append(name)
        rows.append({'job':name,'model':record['model'],'phase':record['phase'],'allocated':a.isoformat(),'release_requested':b.isoformat(),
                     'allocated_seconds_until_release_request':sec,'observed_cu_per_hour':rates[0] if rates else None,
                     'rate_times_seconds_cu':sec*rates[0]/3600 if rates else None,'release_verified':verified,
                     'artifact_verified':bool(record.get('verified_outputs')),'manifest_sha256':digest(path)})
    timeline=sorted(rows,key=lambda q:q['allocated'])
    overlapping=any(v['allocated']<u['release_requested'] for u,v in zip(timeline,timeline[1:]))
    if overlapping:raise ValueError('Overlapping allocations')
    snapshot={'as_of':datetime.now(timezone.utc).isoformat(),'maximum_owned_accelerators_observed':1,
        'checks':'Completed rows use the interval between verified acquisition and the start of release. Colab billing, upload times and minute-rounded usage can differ.',
        'completed_allocation_seconds':sum(x['allocated_seconds_until_release_request'] for x in rows),
        'estimated_compute_units_at_observed_rates':sum(x['rate_times_seconds_cu'] or 0 for x in rows),
        'handled_rates':sum(x['observed_cu_per_hour'] is not None for x in rows),'closed':rows,
        'active_at_last_manifest':pending,'no_accelerator_allocated':no_session,
        'phase_audit': {
            'named_phases':study_receipts, 'specialized_receipts':specialized_archives,
            'allocations_without_final_extraction_receipt':interrupted,
            'unit':'Execution status only. Failed attempts and their successful reruns are separate rows; neither a collected archive nor a completed computation denotes a positive scientific result.'}}
    temporary=DESTINATION.with_suffix('.tmp')
    temporary.write_text(json.dumps(snapshot,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    temporary.replace(DESTINATION)


if __name__=='__main__':ledger()
