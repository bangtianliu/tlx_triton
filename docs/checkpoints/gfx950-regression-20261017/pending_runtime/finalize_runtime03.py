"""Bind CPU constructors first, then admit one exact runtime experiment."""
import argparse
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
PHYSICAL = ROOT.parent / 'physical_plan01'
FROZEN = {'template_READY': {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/runtime_plan01/SOURCE_READY_manifest01.json', 'sha256': 'bd44f992606dc5887f43bb3f9975b46d623416bfef988ece93f15298aa4727ff', 'bytes': 4142}, 'semantic_authorization': {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/post_semantic_admission01.json', 'sha256': 'a6c103ccd26fdfecc30866171fdaa6fc523d2d9c93df2e2cd6b3023830b4f2a8', 'bytes': 8138}, 'plan_template': {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/runtime_plan01/runtime_plan_template01.json', 'sha256': '51f1be2bd0e6eea12b3b4183571b34ca0b7e9174c23ac2aba55b755f6401b9a0', 'bytes': 19930}, 'pipeline_contracts': {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/runtime_plan01/pipeline_contracts_template01.json', 'sha256': '116ba11601df81068705db4945654e711b990455ba784a2582bc38748695bb7f', 'bytes': 299800}, 'pair_template': {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/runtime_plan01/pair_validate_template01.py', 'sha256': 'abff7702907711542dbdbce602b834821acb2dfde01640f895ffa53153f2f97b', 'bytes': 15551}, 'constructor_template': {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/runtime_plan01/constructor_smoke_template02.py', 'sha256': '8761c763c86b9c387e19cbb8fe2a0190e07af8a1b5eb6dc5dcde3d31d628a809', 'bytes': 7940}, 'launcher_template': {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/runtime_plan01/launch_pair_template02.py', 'sha256': '38e96a4908f2f25c7a8d335dae07fd377a4317aa8e2deb447a1864ce4bdc9777', 'bytes': 8568}, 'command_template': {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/runtime_plan01/root_command_template01.json', 'sha256': 'b3cb2fa14e60a64ebb86890d7c171250c7bb57c1294e69a4e2ca4199ba8a1ef6', 'bytes': 7644}}
TEXT_SUFFIXES = {'.py', '.json', '.txt', '.md', '.amdgcn', '.ttir', '.ttgir', '.llir', '.patch', '.td', '.cpp', '.cc'}
COMBINED_CONTRACT = {'path': '/home/bangtliu/meta/tlx_triton/.campaign/pr3993_fragment_spill_validation_20261013_172523/current_pr3993_review_20261014_01/combined_proven_kernel_candidates01/g16_wait_regression_strategy01/paired_Q_coordinate_remat_recovery01/completed_P_Q_dependency_recovery01/mha1_sink_option05/performance_experiment02/runtime_plan01/combined_binding_revision03/binding_contract03.json', 'bytes': 12847, 'sha256': '471e503a23a06ae099c68b927ff04da844822dca31653fb60de6be07799edf31'}


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    require(path.suffix in TEXT_SUFFIXES, 'The binder reads text inputs only')
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def read(record):
    require(isinstance(record, dict) and set(record) == {'path', 'bytes', 'sha256'}, 'A binding remains unresolved')
    require(pin(record['path']) == record, 'A bound text input changed')
    return json.loads(Path(record['path']).read_text())


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return pin(path)


def bind_text(template, target, replacements):
    require(pin(template['path']) == template, 'A source template changed')
    text = Path(template['path']).read_text()
    original = text
    for before, after in replacements:
        require(text.count(before) == 1, 'A finite source binding slot changed')
        text = text.replace(before, after, 1)
    ast.parse(text)
    inverse = text
    for before, after in reversed(replacements):
        require(inverse.count(after) == 1, 'A source inverse slot is ambiguous')
        inverse = inverse.replace(after, before, 1)
    require(inverse == original, 'The source inverse failed')
    with target.open('x') as stream:
        stream.write(text)
    return pin(target)


def check_review(record, schema, expected):
    review = read(record)
    require(review['schema'] == schema and review['passed'] is True, 'An independent source review failed')
    require(review['source_only'] is True and review['findings'] == [], 'The source review has unresolved findings')
    require(review['execution_performed'] is False and review['GPU_admitted'] is False, 'The source review scope changed')
    require(review['checks'] and all(value is True for value in review['checks'].values()), 'A source review check failed')
    for name, value in expected.items():
        require(review['reviewed_files'][name] == value, 'The source review used another file: ' + name)
    for value in review['reviewed_files'].values():
        read_pin = pin(value['path'])
        require(read_pin == value, 'A reviewed file changed')
    return review


def common(inputs):
    for record in FROZEN.values():
        require(pin(record['path']) == record, 'A frozen preparation input changed')
    ready = read(inputs['binder_READY'])
    require(ready['schema'] == 'MHA_SINK_FINITE_BINDER_SOURCE_READY_V1', 'The binder READY schema changed')
    require(ready['binder'] == pin(__file__), 'The finite binder source changed')
    require(ready['template_READY'] == FROZEN['template_READY'], 'The original template READY changed')
    for value in ready['files'].values():
        require(pin(value['path']) == value, 'A binder source input changed')
    initial = read(FROZEN['template_READY'])
    source_pins = read(initial['source_pins'])
    expected = {name: record for name, record in source_pins['outputs'].items()
                if name in ('pipeline_support01.py', 'current_image_factory01.py', 'pair_validate_template01.py',
                            'constructor_smoke_template01.py', 'launch_pair_template01.py', 'runtime_plan_template01.json',
                            'pipeline_contracts_template01.json', 'root_command_template01.json', 'binding_requirements01.json')}
    check_review(inputs['runtime_source_review'], 'MHA_SINK_RUNTIME_TEMPLATE_SOURCE_REVIEW_V1', expected)
    check_review(inputs['binder_source_review'], 'MHA_SINK_RUNTIME_BINDER_SOURCE_REVIEW_V1', ready['files'])
    plan = read(FROZEN['plan_template'])
    require(plan['runtime_ready'] is False and plan['GPU_admission'] is False, 'The source template confers no admission')
    require(plan['original_zero_spill_gate_passed'] is False and plan['production_adoption'] is False, 'Preserve the original rejection')
    semantic = read(FROZEN['semantic_authorization'])
    require(semantic['schema'] == 'ROOT_MHA_SINK_POST_SEMANTIC_AUDIT_AUTHORIZATION_V1', 'The semantic schema changed')
    for key in ('passed', 'bounded_semantic_identity_passed', 'new_contract_resource_gate_passed',
                'original_zero_spill_rejection_preserved', 'physical_audit_authorized'):
        require(semantic[key] is True, 'A semantic authorization gate failed: ' + key)
    require(semantic['GPU_admitted'] is False, 'The semantic authorization remains CPU-only')
    require(plan['frozen_inputs']['post_semantic_authorization'] == FROZEN['semantic_authorization'], 'The semantic binding differs')
    require(semantic['accepted_CPU'] == plan['frozen_inputs']['baseline_CPU']
            and semantic['candidate_CPU'] == plan['frozen_inputs']['candidate_CPU'], 'The CPU authorization differs')
    require(plan['stage'].endswith('/tlx_operand_pin_recovery01/stage'), 'The original T stage is required')
    for record in plan['frozen_inputs'].values():
        require(pin(record['path']) == record, 'A frozen runtime text input changed')
    return plan, semantic


def prepare(inputs, input_pin):
    plan, _ = common(inputs)
    plan['constructor_preparation'] = {'runtime_source_review': inputs['runtime_source_review'],
        'binder_source_review': inputs['binder_source_review'], 'binder_READY': inputs['binder_READY'], 'inputs': input_pin}
    plan['runtime_ready'] = plan['GPU_admission'] = False
    plan_pin = write(ROOT / 'constructor_plan01.json', plan)
    source = bind_text(FROZEN['constructor_template'], ROOT / 'constructor_smoke01.py',
                       [('PENDING_ROOT_BINDING', plan_pin['sha256'])])
    command = {'schema': 'ROOT_MHA_SIX_FROZEN_CONSTRUCTORS_CPU_COMMAND_V1',
        'argv': ['/home/bangtliu/meta_venv/bin/python', '-B', source['path']], 'cwd': str(ROOT),
        'env_updates': {'PYTHONOPTIMIZE': '0', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
                        'HIP_VISIBLE_DEVICES': '-1', 'ROCR_VISIBLE_DEVICES': '-1', 'CUDA_VISIBLE_DEVICES': '-1'},
        'constructor_plan': plan_pin, 'source': source, 'GPU_admission': False,
        'output_must_not_exist': str(ROOT / 'constructor_smoke_result01.json')}
    command_pin = write(ROOT / 'root_constructor_command01.json', command)
    write(ROOT / 'constructor_preparation_receipt01.json', {'schema': 'MHA_SIX_CONSTRUCTOR_PREPARATION_V1',
        'passed': True, 'plan': plan_pin, 'source': source, 'command': command_pin, 'inputs': input_pin,
        'runtime_source_review': inputs['runtime_source_review'], 'binder_source_review': inputs['binder_source_review'],
        'constructor_executed': False, 'GPU_admission': False})


def combined_contract():
    contract = read(COMBINED_CONTRACT)
    require(contract['schema'] == 'MHA_SINK_COMBINED_BINDER_CONTRACT_V3', 'Combined contract schema differs')
    for record in contract['old_files'].values():
        require(pin(record['path']) == record, 'A frozen constructor artifact changed')
    for name in ('combiner_source', 'combination_plan', 'closure_result', 'closure_execution',
                 'closure_independent_review', 'raw_physical_result', 'raw_physical_execution',
                 'raw_memory_readiness', 'raw_native_audit'):
        require(pin(contract[name]['path']) == contract[name], 'A combined contract input changed')
    return contract


def check_constructor_reuse(inputs):
    contract = combined_contract()
    old = contract['old_files']
    require(inputs['constructor_preparation'] == old['constructor_preparation_receipt01.json']
            and inputs['constructor_result'] == old['constructor_smoke_result01.json'], 'Constructor reuse pins differ')
    preparation = read(inputs['constructor_preparation'])
    old_inputs = read(old['root_binding_inputs_prepare01.json'])
    old_ready = read(old['SOURCE_READY_manifest03.json'])
    require(old_ready['binder'] == old['finalize_runtime02.py'], 'Old constructor binder differs')
    require(old_inputs['binder_READY'] == old['SOURCE_READY_manifest03.json']
            and old_inputs['binder_source_review'] == old['old_binder_review'], 'Old preparation bindings differ')
    require(old['old_binder_review']['sha256'] == contract['old_constructor_binder_review_sha256']
            == '69147bc9ca441ea1be26ecd17d25b8c506d747f80c5058c5a658f9c79240f823', 'Old constructor review differs')
    check_review(old['old_binder_review'], 'MHA_SINK_RUNTIME_BINDER_SOURCE_REVIEW_V1', old_ready['files'])
    require(preparation['inputs'] == old['root_binding_inputs_prepare01.json']
            and preparation['binder_source_review'] == old['old_binder_review']
            and preparation['runtime_source_review'] == inputs['runtime_source_review'], 'Constructor preparation provenance differs')
    require(preparation['plan'] == old['constructor_plan01.json']
            and preparation['source'] == old['constructor_smoke01.py']
            and preparation['command'] == old['root_constructor_command01.json'], 'Old constructor source pins differ')
    old_plan = read(preparation['plan'])
    require(old_plan['constructor_preparation'] == {
        'runtime_source_review': preparation['runtime_source_review'],
        'binder_source_review': old['old_binder_review'], 'binder_READY': old['SOURCE_READY_manifest03.json'],
        'inputs': old['root_binding_inputs_prepare01.json']}, 'Old constructor plan provenance differs')
    for name, schema, result_name, command_name in (
        ('root_prepare_execution01.json', 'ROOT_MHA_CONSTRUCTOR_PREPARATION_EXECUTION_V1',
         'constructor_preparation_receipt01.json', 'root_prepare_command01.json'),
        ('root_constructor_execution01.json', 'ROOT_MHA_SIX_CONSTRUCTORS_EXECUTION_V1',
         'constructor_smoke_result01.json', 'root_constructor_execution_command01.json')):
        execution = read(old[name])
        require(execution['schema'] == schema and execution['returncode'] == 0
                and execution['result'] == old[result_name] and execution['command'] == old[command_name]
                and execution['GPU_admitted'] is False, 'Old constructor execution differs')
    old_ast = ast.parse(Path(old['finalize_runtime02.py']['path']).read_text())
    new_ast = ast.parse(Path(__file__).read_text())
    def preparation_AST(tree):
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
        return ast.dump(node, include_attributes=False)
    before, after = preparation_AST(old_ast), preparation_AST(new_ast)
    require(before == after and hashlib.sha256(before.encode()).hexdigest()
            == contract['prepare_function_AST_SHA256'], 'Constructor preparation AST changed')
    return preparation


def check_constructor(inputs, contracts):
    preparation = check_constructor_reuse(inputs)
    require(preparation['schema'] == 'MHA_SIX_CONSTRUCTOR_PREPARATION_V1' and preparation['passed'] is True,
            'Constructor preparation failed')
    require(preparation['plan'] == pin(ROOT / 'constructor_plan01.json')
            and preparation['source'] == pin(ROOT / 'constructor_smoke01.py'), 'Constructor sources changed')
    smoke = read(inputs['constructor_result'])
    require(smoke['passed'] is True and smoke['plan'] == preparation['plan']
            and smoke['check_source'] == preparation['source'], 'The six-constructor receipt differs')
    for key in ('GPU_driver_initialized', 'GPU_handles_initialized', 'framework_imported', 'compiler_pipeline_executed'):
        require(smoke[key] is False, 'A constructor CPU-only gate failed: ' + key)
    require(smoke['original_pack_method_restored'] is True and smoke['live_ASTSource_validated'] is False,
            'Constructor metadata packing or proof scope changed')
    require(set(smoke['six_constructors']) == set(contracts), 'The constructor modes changed')
    expected_hashes = []
    for mode, stages in contracts.items():
        require(set(smoke['six_constructors'][mode]) == set(stages), 'The constructor stages changed')
        for stage, contract in stages.items():
            value, frozen = smoke['six_constructors'][mode][stage], contract['main']
            require(value['name'] == frozen['name'] and value['compile_hash'] == frozen['compile_hash'], 'A constructor shader changed')
            require(value['metadata_exact'] is value['image_exact'] is value['full_options_exact'] is True,
                    'A constructor identity gate failed')
            require(value['saved_launch_kwargs'] == frozen['kwargs']
                    and value['parsed_full_option_dict'] == contract['HIP_options']
                    and value['parsed_external_libraries'] == contract['external_libraries'], 'Constructor options changed')
            expected_hashes.append(frozen['compile_hash'])
    calls = smoke['metadata_packing']
    require(len(calls) == 6 and Counter(row['compile_hash'] for row in calls) == Counter(expected_hashes), 'The constructor set differs')
    require(all(row['legacy_pack'] is False and row['original_pack_method_used'] is True for row in calls),
            'A constructor overrode current metadata packing')
    return preparation


def check_physical(inputs, plan, semantic):
    ready = read(inputs['physical_SOURCE_READY'])
    require(ready['schema'] == 'MHA_SINK_ONE_IMAGE_PHYSICAL_SOURCE_READY_V1', 'The physical source schema differs')
    for value in ready['files']:
        require(pin(value['path']) == value, 'A physical source input changed')
    contract = combined_contract()
    combined = read(inputs['physical_result'])
    require(Path(inputs['physical_result']['path']).resolve() == Path(contract['combined_result_path']),
            'The combined output path differs')
    require(combined['schema'] == 'MHA_SINK_COMBINED_PHYSICAL_READINESS_V2'
            and combined['passed_bounded_checks'] is True and combined['v8_violations'] == 0,
            'The combined physical check failed')
    require(combined['raw_physical_result'] == contract['raw_physical_result']
            and combined['raw_physical_execution'] == contract['raw_physical_execution']
            and combined['raw_memory_readiness'] == contract['raw_memory_readiness']
            and combined['saved_row_closure'] == contract['closure_result']
            and combined['closure_execution'] == contract['closure_execution']
            and combined['closure_independent_review'] == contract['closure_independent_review'],
            'The raw failure and closure provenance differ')
    require(combined['raw_passed_bounded_checks'] is False and combined['raw_returncode'] == 1
            and combined['raw_readiness_counts'] == {'LDS': {'proofs': 2426, 'unresolved': 0},
                                                   'VMEM': {'proofs': 308, 'unresolved': 81}},
            'The combined result changed the raw failure')
    require(combined['readiness_counts'] == {'LDS': {'proofs': 2426, 'unresolved': 0},
                                           'VMEM': {'proofs': 389, 'unresolved': 0}}
            and combined['closed_saved_row_count'] == 81 and combined['closure_negative_control_count'] == 5,
            'The combined closure counts differ')
    require(combined['combiner_source'] == contract['combiner_source']
            and combined['combination_plan'] == contract['combination_plan']
            and combined['combination_source_review'] == inputs['binder_source_review']
            and combined['combination_command'] == inputs['physical_command'], 'Combined source bindings differ')
    require(combined['aggregation_only'] is True and combined['audit_performed'] is False
            and combined['physical_audit_count'] == 1 and combined['native_v8_rerun'] is False
            and combined['original_native_parser_rerun'] is False
            and combined['GPU_admitted'] is False and combined['runtime_admitted'] is False,
            'Combined aggregation scope differs')
    physical = read(combined['raw_physical_result'])
    require(Path(combined['raw_physical_result']['path']).resolve() == PHYSICAL / 'physical_result01/root_physical_readiness01.json',
            'The physical output path differs')
    require(physical['schema'] == 'MHA_SINK_ONE_IMAGE_PHYSICAL_READINESS_V1'
            and physical['passed_bounded_checks'] is False and physical['v8_violations'] == 0, 'The raw physical result changed')
    require(physical['CPU'] == plan['frozen_inputs']['candidate_CPU'] and physical['assembly'] == semantic['assembly']
            and physical['candidate_image_reference_only'] == semantic['image'], 'The physical image binding differs')
    require(physical['baseline_audit'] is False and physical['audit_count_limit'] == 1
            and physical['original_zero_spill_rejection_preserved'] is True
            and physical['runtime_admitted'] is False and physical['GPU_admitted'] is False, 'The physical scope changed')
    binding_record = physical['bound_inputs']['bindings']
    bindings = read(binding_record)
    require(bindings['post_semantic_admission'] == FROZEN['semantic_authorization'], 'The physical semantic authorization differs')
    keys = ('post_semantic_admission', 'experiment_contract', 'candidate_CPU', 'accepted_CPU_reference_only',
            'candidate_source', 'candidate_assembly', 'stage_identity', 'stage_contract', 'independent_result_review',
            'original_failed_disposition', 'native_audit_v8', 'original_physical_adapter')
    expected = {key: bindings[key] for key in keys}
    expected.update(adapter=physical['bound_inputs']['adapter'], bindings=binding_record)
    require(physical['bound_inputs'] == expected, 'The physical input closure differs')
    require(expected['adapter'] in ready['files'], 'The physical adapter was not in its READY manifest')
    for value in expected.values():
        require(pin(value['path']) == value, 'A physical bound input changed')
    require(physical['readiness_counts'] == combined['raw_readiness_counts'], 'Raw readiness counts differ')
    ledger = read(combined['memory_readiness'])
    raw_memory = read(physical['memory_readiness'])
    closure = read(contract['closure_result'])
    require(ledger['schema'] == 'MHA_SINK_COMBINED_MEMORY_LEDGER_V2'
            and ledger['raw_memory_readiness'] == physical['memory_readiness']
            and ledger['saved_row_closure'] == contract['closure_result']
            and ledger['original_proofs_preserved'] is True and ledger['closed_saved_row_count'] == 81,
            'Combined memory ledger provenance differs')
    require(ledger['LDS'] == raw_memory['LDS'] and ledger['VMEM']['proofs'][:308] == raw_memory['VMEM']['proofs']
            and ledger['VMEM']['unresolved'] == [] and len(ledger['VMEM']['proofs']) == 389,
            'Original memory proofs changed')
    require(ledger['VMEM']['proofs'][308:] == [dict(row['saved_row_exact'], valid_wait_lines=[row['wait_line']])
            for row in closure['proofs']], 'Closed saved rows differ')
    require(combined['audit'] == physical['audit'] == contract['raw_native_audit']
            and combined['CPU'] == physical['CPU'] and combined['assembly'] == physical['assembly']
            and combined['candidate_image_reference_only'] == physical['candidate_image_reference_only'],
            'Combined image or native audit differs')
    require(read(physical['audit'])['violations'] == [], 'Raw native audit has violations')
    for key in ('native_v8', 'audit', 'memory_readiness'):
        require(pin(physical[key]['path']) == physical[key], 'A physical output changed')
    reservation_path = PHYSICAL / 'physical_result01/root_audit_reservation01.json'
    reservation = json.loads(reservation_path.read_text())
    require(reservation['schema'] == 'MHA_SINK_ONE_IMAGE_AUDIT_RESERVATION_V1'
            and reservation['root_admission'] == FROZEN['semantic_authorization']
            and reservation['candidate_image_reference_only'] == semantic['image']
            and reservation['audit_count_limit'] == 1 and reservation['baseline_audit'] is False, 'The audit reservation differs')
    require(len(list(PHYSICAL.glob('physical_result*/root_audit_reservation*.json'))) == 1,
            'The new-image audit reservation count differs')
    require(len(list(PHYSICAL.glob('physical_result*/candidate_native_v8_audit01.json'))) == 1,
            'The new-image audit count differs')
    execution = read(combined['raw_physical_execution'])
    require(execution['schema'] == 'ROOT_MHA_SINK_ONE_IMAGE_AUDIT_EXECUTION_V1' and execution['returncode'] == 1,
            'The raw physical execution changed')
    require(execution['command'] == pin(PHYSICAL / 'root_command01.json')
            and execution['result'] == combined['raw_physical_result'],
            'The root physical execution pins differ')
    command = read(execution['command'])
    require(command['argv'] == ['/usr/bin/python3', '-B', expected['adapter']['path'],
                               '--root-admission-sha256', FROZEN['semantic_authorization']['sha256']],
            'The actual physical command differs')
    require(command['cwd'] == str(PHYSICAL), 'The physical command directory differs')
    aggregation_execution = read(inputs['physical_execution'])
    require(aggregation_execution['schema'] == contract['combined_execution_schema']
            and aggregation_execution['returncode'] == 0
            and aggregation_execution['command'] == inputs['physical_command']
            and aggregation_execution['result'] == inputs['physical_result'], 'Root aggregation execution differs')
    require(aggregation_execution['aggregation_only'] is True and aggregation_execution['audit_performed'] is False
            and aggregation_execution['physical_audit_count'] == 1
            and aggregation_execution['GPU_admitted'] is False, 'Root aggregation scope differs')
    command = read(inputs['physical_command'])
    require(Path(inputs['physical_command']['path']).resolve() == Path(contract['combined_command_path'])
            and command['schema'] == 'ROOT_MHA_COMBINED_READINESS_COMMAND_V2'
            and command['source'] == contract['combiner_source'] and command['plan'] == contract['combination_plan']
            and command['source_review'] == inputs['binder_source_review'], 'Aggregation command bindings differ')
    require(command['argv'] == ['/usr/bin/python3', '-B', contract['combiner_source']['path'],
            '--plan-sha256', contract['combination_plan']['sha256'],
            '--source-sha256', contract['combiner_source']['sha256'],
            '--source-review', inputs['binder_source_review']['path'],
            '--source-review-sha256', inputs['binder_source_review']['sha256']], 'Aggregation arguments differ')
    require(command['cwd'] == str(Path(contract['combiner_source']['path']).parent)
            and command['aggregation_only'] is True, 'Aggregation command scope differs')
    review = read(inputs['physical_result_review'])
    require(review['schema'] == contract['combined_result_review_schema'] and review['passed'] is True
            and review['findings'] == [] and review['physical_result'] == inputs['physical_result']
            and review['root_execution'] == inputs['physical_execution']
            and review['raw_physical_result'] == contract['raw_physical_result']
            and review['saved_row_closure'] == contract['closure_result'], 'Independent combined review differs')
    return combined


def check_quiet(record, expected):
    quiet = read(record)
    require(quiet['schema'] == 'ROOT_MHA_FRESH_QUIET_WINDOW_V1' and quiet['passed'] is True,
            'A fresh root quiet window is required')
    require(quiet['physical_GPU'] == expected and len(quiet['samples']) >= 2, 'The quiet GPU binding differs')
    timestamps = [sample['unix'] for sample in quiet['samples']]
    require(all(type(value) in (int, float) for value in timestamps), 'Quiet timestamps are invalid')
    require(all(left < right for left, right in zip(timestamps, timestamps[1:])), 'Quiet samples are not ordered')
    require(timestamps[-1] - timestamps[0] >= 1 and 0 <= time.time() - timestamps[-1] <= 30,
            'The quiet window is stale')
    for sample in quiet['samples']:
        rows = [row for row in sample['inventory']['GPUs'] if row['physical'] == expected['index']]
        require(len(rows) == 1, 'The quiet GPU inventory is ambiguous')
        row = rows[0]
        require(row['quiet'] is True and row['BDF'] == expected['BDF']
                and row['KFD_id'] == expected['KFD_id'] and row['HIP_ordinal'] == expected['HIP_ordinal'],
                'A quiet sample failed the exact GPU mapping')


def finalize(inputs):
    plan, semantic = common(inputs)
    contracts = read(FROZEN['pipeline_contracts'])
    preparation = check_constructor(inputs, contracts)
    combined = check_physical(inputs, plan, semantic)
    check_quiet(inputs['fresh_quiet_window'], plan['physical_GPU'])
    new_bindings = {'physical_readiness': inputs['physical_result'], 'root_physical_execution': inputs['physical_execution'],
        'independent_physical_review': inputs['physical_result_review'], 'independent_runtime_source_review': inputs['runtime_source_review']}
    admission = {'schema': 'ROOT_MHA_SINK_RELOAD_RUNTIME_ADMISSION_V1', 'passed': True, 'approved_for_runtime': True,
        'GPU_admitted': True, 'post_semantic_authorization': FROZEN['semantic_authorization'],
        'baseline_CPU': plan['frozen_inputs']['baseline_CPU'], 'candidate_CPU': plan['frozen_inputs']['candidate_CPU'],
        'recorded_main_images': plan['recorded_main_images'], 'required_new_bindings': new_bindings,
        'physical_readiness_passed': True, 'physical_audit_count': 1, 'baseline_physical_audit_count': 0,
        'raw_physical_result': combined['raw_physical_result'],
        'raw_physical_execution': combined['raw_physical_execution'],
        'saved_row_closure': combined['saved_row_closure'], 'closure_execution': combined['closure_execution'],
        'aggregation_only_physical_revision': True, 'constructors_reused': True,
        'constructor_plan': preparation['plan'], 'constructor_result': inputs['constructor_result'],
        'binder_source_review': inputs['binder_source_review'], 'fresh_quiet_window': inputs['fresh_quiet_window'],
        'original_zero_spill_gate_passed': False, 'production_adoption': False, 'performance_gain_proven': False,
        'final_binding_review_required': True, 'fresh_launcher_quiet_and_sampled_closure_required': True}
    admission_pin = write(ROOT / 'root_runtime_admission01.json', admission)
    plan['required_new_bindings'] = dict(new_bindings, root_runtime_admission=admission_pin)
    plan.update(runtime_ready=True, GPU_admission=True, constructor_plan=preparation['plan'],
                constructor_result=inputs['constructor_result'], fresh_quiet_window=inputs['fresh_quiet_window'])
    plan_pin = write(ROOT / 'runtime_plan_bound01.json', plan)
    adapter = bind_text(FROZEN['pair_template'], ROOT / 'pair_validate01.py',
        [('FINALIZED = False', 'FINALIZED = True'), ('PENDING_ROOT_BINDING', plan_pin['sha256'])])
    launcher = bind_text(FROZEN['launcher_template'], ROOT / 'launch_pair01.py',
        [('FINALIZED = False', 'FINALIZED = True'), ("PLAN_SHA = 'PENDING_ROOT_BINDING'", "PLAN_SHA = '" + plan_pin['sha256'] + "'"),
         ("DRIVER_SHA = 'PENDING_ROOT_BINDING'", "DRIVER_SHA = '" + adapter['sha256'] + "'")])
    command = read(FROZEN['command_template'])
    command.update(runtime_ready=True, GPU_admission=False, final_binding_review_pending=True,
                   adapter=adapter, launcher=launcher, plan=plan_pin, runtime_admission=admission_pin)
    command_pin = write(ROOT / 'root_command_bound_template01.json', command)
    write(ROOT / 'final_binding_preparation_receipt01.json', {'schema': 'MHA_SINK_FINITE_BINDING_PREPARATION_V1',
        'passed': True, 'plan': plan_pin, 'adapter': adapter, 'launcher': launcher, 'command_template': command_pin,
        'runtime_admission': admission_pin, 'constructor_result': inputs['constructor_result'],
        'physical_result': inputs['physical_result'], 'fresh_quiet_window': inputs['fresh_quiet_window'],
        'GPU_execution_performed': False, 'final_binding_review_pending': True, 'timing_qualified': False,
        'production_adoption': False, 'original_zero_spill_gate_passed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare-constructors', 'finalize'))
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--bindings-sha256', required=True)
    args = parser.parse_args()
    import sys
    require(sys.flags.optimize == 0, 'Keep Python assertions enabled')
    record = pin(args.bindings)
    require(record['sha256'] == args.bindings_sha256, 'The root binding inputs changed')
    inputs = read(record)
    require(inputs['schema'] == 'ROOT_MHA_SINK_FINITE_BINDING_INPUTS_V1', 'The binding input schema differs')
    require(args.phase == 'finalize', 'Reuse the completed constructor preparation')
    if args.phase == 'prepare-constructors':
        prepare(inputs, record)
    else:
        finalize(inputs)


if __name__ == '__main__':
    main()
