"""Aggregate the pinned raw audit and saved-row closure. Do not run an audit."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT = {'.json', '.py', '.md', '.amdgcn', '.cpp', '.h', '.td', '.patch'}
RAW_COUNTS = {'LDS': {'proofs': 2426, 'unresolved': 0}, 'VMEM': {'proofs': 308, 'unresolved': 81}}
COMBINED_COUNTS = {'LDS': {'proofs': 2426, 'unresolved': 0}, 'VMEM': {'proofs': 389, 'unresolved': 0}}
NEGATIVES = {'missing_wait', 'insufficient_younger_load_count', 'wrong_producer',
             'altered_text_binding', 'altered_image_binding'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    require(path.suffix in TEXT, 'Read text inputs only')
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def verify(record):
    require(pin(record['path']) == record, 'Pinned text changed')


def read(record):
    verify(record)
    require(Path(record['path']).suffix == '.json', 'Require JSON input')
    return json.loads(Path(record['path']).read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return pin(path)


def review_passed(review, schema):
    require(review['schema'] == schema and review['passed'] is True and review['source_only'] is True,
            'Independent source review failed')
    require(review['findings'] == [] and review['execution_performed'] is False
            and review['GPU_admitted'] is False, 'Independent source review scope changed')
    require(review['checks'] and all(value is True for value in review['checks'].values()),
            'Independent source review has a failed check')
    for record in review['reviewed_files'].values():
        verify(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--source-review-sha256', required=True)
    args = parser.parse_args()
    source = pin(__file__)
    plan_pin = pin(HERE / 'combination_plan01.json')
    require(source['sha256'] == args.source_sha256 and plan_pin['sha256'] == args.plan_sha256,
            'Root source authorization differs')
    plan = read(plan_pin)
    require(plan['schema'] == 'MHA_SINK_COMBINED_READINESS_PLAN_V2', 'Wrong aggregation plan')
    review_pin = pin(args.source_review)
    require(review_pin['sha256'] == args.source_review_sha256, 'Root source review binding differs')
    source_review = read(review_pin)
    review_passed(source_review, 'MHA_SINK_RUNTIME_BINDER_SOURCE_REVIEW_V1')
    require(source_review['reviewed_files']['combine_readiness01.py'] == source
            and source_review['reviewed_files']['combination_plan01.json'] == plan_pin,
            'Source review binds another aggregation')
    command_pin = pin(HERE / 'combined_root_command01.json')
    command = read(command_pin)
    require(command['schema'] == 'ROOT_MHA_COMBINED_READINESS_COMMAND_V2'
            and command['source'] == source and command['plan'] == plan_pin
            and command['source_review'] == review_pin, 'Aggregation command bindings differ')
    require(command['argv'] == ['/usr/bin/python3', '-B', source['path'], '--plan-sha256', args.plan_sha256,
                                '--source-sha256', args.source_sha256, '--source-review', review_pin['path'],
                                '--source-review-sha256', review_pin['sha256']], 'Aggregation arguments differ')
    require(command['cwd'] == str(HERE) and command['aggregation_only'] is True, 'Aggregation command scope differs')
    inputs = plan['inputs']
    for record in inputs.values():
        verify(record)
    raw = read(inputs['raw_physical_result'])
    memory = read(inputs['raw_memory_readiness'])
    native = read(inputs['raw_native_audit'])
    raw_execution = read(inputs['raw_physical_execution'])
    closure = read(inputs['closure_result'])
    execution = read(inputs['closure_execution'])
    closure_plan = read(inputs['closure_plan'])
    closure_review = read(inputs['closure_independent_review'])
    review_passed(closure_review, 'MHA_SINK_SAVED_VMEM_CLOSURE_SOURCE_REVIEW_V1')
    require(closure_review['reviewed_files'] == plan['closure_reviewed_files'], 'Closure review file set differs')
    require(raw['schema'] == 'MHA_SINK_ONE_IMAGE_PHYSICAL_READINESS_V1'
            and raw['passed_bounded_checks'] is False and raw['v8_violations'] == 0
            and raw['readiness_counts'] == RAW_COUNTS, 'Original failed result changed')
    require(raw_execution['schema'] == 'ROOT_MHA_SINK_ONE_IMAGE_AUDIT_EXECUTION_V1'
            and raw_execution['returncode'] == 1 and raw_execution['result'] == inputs['raw_physical_result'],
            'Original failed execution changed')
    require(raw['audit'] == inputs['raw_native_audit'] and raw['memory_readiness'] == inputs['raw_memory_readiness'],
            'Raw evidence bindings differ')
    require(native['violations'] == [] and native['accum_offset'] == 256
            and native['sha256'] == raw['assembly']['sha256'], 'Raw native evidence differs')
    require({kind: {name: len(items) for name, items in data.items()}
             for kind, data in memory.items()} == RAW_COUNTS, 'Raw memory counts differ')
    require(raw['baseline_audit'] is False and raw['audit_count_limit'] == 1
            and raw['GPU_admitted'] is False and raw['runtime_admitted'] is False, 'Raw audit scope differs')
    require(closure['schema'] == 'MHA_SINK_SAVED_VMEM_ROW_CLOSURE_V1'
            and closure['passed_saved_row_closure'] is True
            and closure['saved_unresolved_row_count'] == closure['closed_row_count'] == 81
            and closure['unresolved_row_count'] == 0, 'Saved-row closure failed')
    pairs = {'original_readiness': 'raw_memory_readiness', 'original_native_audit': 'raw_native_audit',
             'original_root_execution': 'raw_physical_execution',
             'original_root_physical_readiness': 'raw_physical_result', 'source': 'closure_source',
             'pinned_plan': 'closure_plan', 'root_command': 'closure_command'}
    require(all(closure[key] == inputs[name] for key, name in pairs.items()), 'Closure input bindings differ')
    require(closure['original_returncode'] == 1 and closure['original_passed_bounded_checks'] is False
            and closure['inherited_readiness_counts'] == RAW_COUNTS
            and closure['combined_readiness_counts'] == COMBINED_COUNTS, 'Closure raw provenance differs')
    require(execution['schema'] == 'ROOT_MHA_SAVED_VMEM_CLOSURE_EXECUTION_V1'
            and execution['returncode'] == 0 and execution['command'] == inputs['closure_command']
            and execution['result'] == inputs['closure_result']
            and execution['independent_review'] == inputs['closure_independent_review'], 'Closure root execution differs')
    require(execution['audit_rerun'] is False and execution['physical_audit_count'] == 1
            and execution['GPU_admitted'] is False, 'Closure execution scope differs')
    require(closure['physical_audit_count'] == 1 and all(closure[key] is False for key in
            ('physical_audit_rerun', 'native_v8_rerun', 'original_native_parser_rerun', 'binary_payloads_read',
             'compiler_run', 'framework_imported', 'GPU_admitted', 'runtime_admitted', 'performance_gain_proven')),
            'Closure proof scope differs')
    require(closure_review['reviewed_files']['close_saved_rows01.py'] == closure['source']
            and closure_review['reviewed_files']['closure_plan01.json'] == closure['pinned_plan'],
            'Closure review binds other proof sources')
    for record in closure_plan['inputs'].values():
        verify(record)
    for record in closure_plan['primary_source_files'].values():
        verify(record)
    require(closure_plan['saved_rows'] == memory['VMEM']['unresolved'], 'Closure plan rows differ')
    controls = closure['negative_controls']
    require(len(controls) == 5 and {item['name'] for item in controls} == NEGATIVES
            and all(item['passed'] is True and item['rejection'] for item in controls), 'Closure negative controls differ')
    require(closure['candidate_assembly'] == raw['assembly']
            and closure['candidate_image_reference_only'] == raw['candidate_image_reference_only']
            == closure_plan['candidate_image_reference_only'], 'Closure image binding differs')
    require(raw['assembly'] == closure_plan['inputs']['candidate_assembly'], 'Closure text binding differs')
    bindings = read(raw['bound_inputs']['bindings'])
    cpu = read(bindings['candidate_CPU'])
    semantic = read(bindings['post_semantic_admission'])
    require(raw['assembly'] == bindings['candidate_assembly'] == semantic['assembly']
            == cpu['programs']['main']['files']['amdgcn'], 'Candidate text identity differs')
    require(raw['candidate_image_reference_only'] == bindings['candidate_image_reference_only'] == semantic['image']
            == cpu['programs']['main']['files']['hsaco'], 'Candidate image identity differs')
    require(cpu['programs']['main']['loader_payload_SHA256'] == semantic['image']['sha256'], 'Loader image identity differs')
    require(raw['assembly']['sha256'] == '4184d449a8a9ea8ad1e9b03ecf7445415bc55150cfb7b415975d00cce8d9774b'
            and semantic['image']['sha256'] == '2bbd14a99efdf208b2662d8b5dbc9a6f1293b936d2ad50f44db446ae72d66221',
            'Unsupported candidate image')
    proofs = closure['proofs']
    require(len(proofs) == 81, 'Closure proof count differs')
    closed = []
    for index, (proof, row) in enumerate(zip(proofs, memory['VMEM']['unresolved'])):
        require(proof['saved_row_index'] == index and proof['saved_row_exact'] == row
                and proof['resolved'] is True and proof['wait_dominates_consumer'] is True
                and proof['producer_and_events_reach_wait_without_branch'] is True
                and proof['producer_definition_preserved'] is True, 'Closed row identity differs')
        require(proof['younger_event_count'] == len(proof['younger_event_lines'])
                and proof['younger_event_count'] >= proof['vmcnt_threshold'], 'Closed row count differs')
        closed.append(dict(row, valid_wait_lines=[proof['wait_line']]))
    ledger = {'schema': 'MHA_SINK_COMBINED_MEMORY_LEDGER_V2',
              'LDS': memory['LDS'], 'VMEM': {'proofs': memory['VMEM']['proofs'] + closed, 'unresolved': []},
              'raw_memory_readiness': inputs['raw_memory_readiness'], 'saved_row_closure': inputs['closure_result'],
              'original_proofs_preserved': True, 'closed_saved_row_count': 81}
    require(ledger['LDS'] == memory['LDS'] and ledger['VMEM']['proofs'][:308] == memory['VMEM']['proofs'],
            'Original memory proofs changed')
    for record in inputs.values():
        verify(record)
    verify(source)
    verify(plan_pin)
    verify(review_pin)
    verify(command_pin)
    output = HERE / 'combined_result01'
    require(not output.exists(), 'Preserve aggregation output')
    output.mkdir(exist_ok=False)
    ledger_pin = write(output / 'combined_memory_readiness01.json', ledger)
    combined = dict(raw)
    combined.update(schema='MHA_SINK_COMBINED_PHYSICAL_READINESS_V2', passed_bounded_checks=True,
        readiness_counts=COMBINED_COUNTS, memory_readiness=ledger_pin, combined_memory_ledger=ledger_pin,
        raw_physical_result=inputs['raw_physical_result'], raw_physical_execution=inputs['raw_physical_execution'],
        raw_memory_readiness=inputs['raw_memory_readiness'], raw_passed_bounded_checks=False, raw_returncode=1,
        raw_readiness_counts=RAW_COUNTS, saved_row_closure=inputs['closure_result'],
        closure_execution=inputs['closure_execution'], closure_independent_review=inputs['closure_independent_review'],
        closed_saved_row_count=81, closure_negative_control_count=5, combiner_source=source,
        combination_plan=plan_pin, combination_source_review=review_pin, combination_command=command_pin,
        aggregation_only=True, audit_performed=False, physical_audit_count=1,
        native_v8_rerun=False, original_native_parser_rerun=False,
        limits=['This result combines the original audit and the exact saved-row closure.',
                'The original audit result and its failed execution remain unchanged.',
                'This result does not prove general asynchronous safety, numerical correctness, or performance.'])
    result = write(output / 'root_combined_readiness01.json', combined)
    print(json.dumps({'passed': True, 'aggregation_only': True, 'result': result, 'memory_readiness': ledger_pin}))


if __name__ == '__main__':
    main()
