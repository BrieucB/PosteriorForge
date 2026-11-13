"""Pair-bound FP32 routing-v2 numerical screen. Built with PriorLabs-TabPFN.

Reuses exact922 collector/numeric primitives in a NEW private namespace. The
collector AST edits rebind the v2 backend and fixed seed/case to an immutable
PairSpec. Inputs, custody and output identity are explicitly new plumbing.
No frozen global dictionary, model/codec math, masks or thresholds is modified.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,inspect,json
from pathlib import Path
from types import FunctionType,SimpleNamespace
from posteriorforge import prior_coordinate_pair_v1 as pairs
from posteriorforge.artifacts import write_json

SCHEMA="posteriorforge.prior_coordinate_pair_conditionals.v2"
PASS_STATUS="prior_coordinate_pair_v2_screen_pass"
NUMERIC_HELPER_SHA256="9221da7853d1ecea3606d9001c70589f61bb89a28717bbc333bc8a280ff81bf9"
FIFO_SHA256="f8823ae75e7ae2eea6f1aa667af48ecf5c892450361b5abd490ccda36a28f7e0"
SELECTION_ANCHOR_SHA256="5ec69eb7aadbd74c7d96ee534b71d0d8bee2b6dd6f6e3d2f44e74607452f682b"
PAIR_MODULE_SHA256="402117cad65af190352cb0b14977fbbdc896f196d069749b0be28608fdef400e"
PAIR_CPU_HELPER_SHA256="b69a92b59fa504bce22e34dc31895f9162870425118764293f959c46390110e4"
PAIR_CPU_RECEIPT_SHA256="6f35d9d090a955604186c5fbf3312b82465eff6fdae7e4cd24a23b646dbdf193"
V1_QUALIFIER_SHA256="29ca88203b81f240769a6bb38b18b68aff0fe6ed2df36003729b17e0116163a0"
V2_BACKEND_SHA256="0199b16e9cc34203fa0d5645d161742fe9ed244633d7beee047fbcbf2528661c"
V1_PAIR_FAILURE_SHA256="fbed99d9fb256bb72a49cb03bebb820d805ac2cbafaa4ebac4deb08bf3160734"
V2_PREREGISTRATION_SHA256="8640a4e157698862823521d3fd14396ecc3ebf121dffffabc987cc937d3ad2fc"
original_numeric=pairs.load_script("qualify_prior_native_conditionals_v1.py",NUMERIC_HELPER_SHA256)


def _backend_imports(tree,reverse=False):
    old,new=('execution_prior_native_v2','execution_prior_native_v1') if reverse else ('execution_prior_native_v1','execution_prior_native_v2')
    count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module=='posteriorforge':
            for alias in node.names:
                if alias.name==old:
                    pairs.require(alias.asname=='native','native backend alias changed');alias.name=new;count+=1
    pairs.require(count==1,'one exact local v2 backend import required')
    return count


def _make_numeric_namespace():
    namespace=dict(vars(original_numeric));namespace.update(SCHEMA=SCHEMA,NATIVE_STAGES=(6,10,14))
    witness=original_numeric.witness_guard
    namespace['witness_guard']=FunctionType(witness.__code__,namespace,witness.__name__,witness.__defaults__,witness.__closure__)
    namespace['witness_guard'].__kwdefaults__=witness.__kwdefaults__
    original=ast.parse(inspect.getsource(original_numeric.validate_screen)).body[0];candidate=copy.deepcopy(original)
    _backend_imports(candidate);changes={'native_all8_stages':0,'cached_stages':0}
    for node in ast.walk(candidate):
        if isinstance(node,ast.Dict):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='native_all8_stages':
                    pairs.require(ast.literal_eval(node.values[i])==[6,10],'original native-stage return changed');node.values[i]=ast.parse('[6,10,14]',mode='eval').body;changes[key.value]+=1
                if isinstance(key,ast.Constant) and key.value=='cached_stages':
                    pairs.require(ast.literal_eval(node.values[i])==19,'original cached-count return changed');node.values[i]=ast.Constant(18);changes[key.value]+=1
    pairs.require(all(v==1 for v in changes.values()),'exact validator count metadata deltas required')
    reverse=copy.deepcopy(candidate);_backend_imports(reverse,True)
    for node in ast.walk(reverse):
        if isinstance(node,ast.Dict):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='native_all8_stages':node.values[i]=ast.parse('[6,10]',mode='eval').body
                if isinstance(key,ast.Constant) and key.value=='cached_stages':node.values[i]=ast.Constant(19)
    pairs.require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),'undeclared v2 numeric validator AST change')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[])),__file__+'::private922_validator_backend_metadata_v2','exec'),namespace)
    # Keep original550 selection at(6,10), never reinterpret it under the v2 tuple.
    namespace['failed_screen_receipt']=original_numeric.failed_screen_receipt
    return SimpleNamespace(**namespace)


numeric=_make_numeric_namespace()
witness_guard=numeric.witness_guard
stage_decision=numeric.stage_decision
THRESHOLDS=numeric.THRESHOLDS


def source_sha256():return pairs.sha(__file__)


def source_pins():
    pairs.require(all(isinstance(x,str) and len(x)==64 for x in (PAIR_MODULE_SHA256,PAIR_CPU_HELPER_SHA256,PAIR_CPU_RECEIPT_SHA256)),"pair sources/dualCPU receipt not frozen; GPU screen disabled")
    pairs.require(pairs.source_sha256()==PAIR_MODULE_SHA256,"frozen PairSpec/input source changed")
    root=Path(__file__).resolve().parents[1]
    pairs.require(pairs.sha(root/'scripts/preflight_prior_coordinate_pairs_v1.py')==PAIR_CPU_HELPER_SHA256,"fixed CPU preflight source changed")
    from posteriorforge import execution_physical_fifo_v1 as fifo
    pairs.require(fifo.source_sha256()==FIFO_SHA256,"fixed physicalF64 FIFO changed")
    pairs.require(pairs.sha(root/"scripts/qualify_prior_native_conditionals_v1.py")==NUMERIC_HELPER_SHA256,"frozen numerical collector source changed")
    pairs.require(pairs.sha(root/'scripts/qualify_prior_coordinate_pair_v1.py')==V1_QUALIFIER_SHA256,'frozen pair-v1 qualifier changed')
    from posteriorforge import execution_prior_native_v2 as backend
    pairs.require(backend.source_sha256()==V2_BACKEND_SHA256,'frozen v2 backend changed');backend.require_sources()
    pairs.require_sources()
    return {"qualifier":source_sha256(),"pair_inputs":PAIR_MODULE_SHA256,"pair_CPU_helper":PAIR_CPU_HELPER_SHA256,
        "pair_CPU_receipt":PAIR_CPU_RECEIPT_SHA256,"numerical_collector_ancestry":NUMERIC_HELPER_SHA256,
        "physical_FIFO":FIFO_SHA256,"pair_sources":pairs.source_pins(),"numeric_ancestral_sources":original_numeric.source_pins(),"fixed_selection_anchor":SELECTION_ANCHOR_SHA256,"execution_v2":V2_BACKEND_SHA256,"v2_preregistration":V2_PREREGISTRATION_SHA256,"v1_case0_failure_selection":V1_PAIR_FAILURE_SHA256,"pair_v1_qualifier_ancestry":V1_QUALIFIER_SHA256,"CPU_and_case2_v1_are_data_ancestry_only":True}


def _cpu(path,spec):
    pairs.require(isinstance(PAIR_CPU_RECEIPT_SHA256,str) and pairs.sha(path)==PAIR_CPU_RECEIPT_SHA256,"exact all23 dualCPU receipt required")
    doc=json.loads(Path(path).read_bytes());row=pairs.validate_pair_cpu_receipt(doc,spec)
    pairs.require(doc.get("producer_source_sha256")==PAIR_CPU_HELPER_SHA256 and doc.get("case2_regression_passed_both_environments") is True,"CPU source/case2 regression differs")
    return doc,row


def _selection(path,anchor_path,v1_pair_failure):
    anchor=pairs._document(anchor_path,SELECTION_ANCHOR_SHA256)
    result=numeric.failed_screen_receipt(path,anchor['raw_bindings'],anchor['transformed_bindings'])
    failure=pairs._document(v1_pair_failure,V1_PAIR_FAILURE_SHA256)
    pairs.require(failure['schema']=='posteriorforge.prior_coordinate_pair_conditionals.v1' and failure['pair']==pairs.FIRST_PAIR.as_dict()
        and failure['FP32_failed_stages']==[14] and all(r['hard_validity_passed'] for r in failure['stage_results'])
        and failure['sources']['qualifier']==V1_QUALIFIER_SHA256,'exact v1 case0 numerical-failure selection required')
    return {**result,"scope":"original550 selects6/10; v1 actualcase0 failure selects14; selection only, never v2 proof","qualifies_actual_pair":False,
        "v1_case0_selection":{"manifest_sha256":V1_PAIR_FAILURE_SHA256,"qualification_passed":False,"source_status_preserved":failure['status'],"failed_stages":[14]},"predeclared_v2_native_all8_stages":[6,10,14]}


def _inventory_replay(base,spec):
    return {"training_inventory":base,"pair":spec.as_dict(),"pair_CPU_receipt_sha256":PAIR_CPU_RECEIPT_SHA256,
        "COMMON_Float32_model_bytes_match":True,"Float64_cross_environment_byte_identity_claimed":False,
        "scope":"existing three-bank codec proof plus new actual-pair fixed-prefix dualCPU preflight only"}


def _numeric_validate(document,**kwargs):
    namespace=dict(numeric.validate_screen.__globals__);namespace['SCHEMA']=SCHEMA
    clone=FunctionType(numeric.validate_screen.__code__,namespace,numeric.validate_screen.__name__,numeric.validate_screen.__defaults__,numeric.validate_screen.__closure__)
    clone.__kwdefaults__=numeric.validate_screen.__kwdefaults__
    return clone(document,**kwargs)


def validate_screen(document,*,spec,expected_sources,raw_bindings,transformed_bindings,coordinate_policy,native_policy,inventory_replay,pair_cpu_receipt):
    pairs.require(document.get('schema')==SCHEMA and document.get('status')==PASS_STATUS,"new pair proof schema/pass status required; old5ec cannot qualify another pair")
    pairs.require(document.get('pair')==spec.as_dict() and document.get('fixed_panel_preregistration_sha256')==pairs.PREREGISTRATION_SHA256,"actual proof PairSpec/prereg differs")
    pairs.require(document.get('pair_CPU_receipt_sha256')==PAIR_CPU_RECEIPT_SHA256,"proof must bind frozen all23 CPU receipt")
    row=pairs.validate_pair_cpu_receipt(pair_cpu_receipt,spec)
    pairs.require(document.get('prefix_custody')==row['inference']['prefix_custody'],"actual FIRST1024 pure-parent custody differs")
    pairs.require([r['mask_inputs'] for r in document.get('stage_results',[])]==row['inference']['all21_mask_target_query_bindings'],"actual pair all21 masks/targets/query bytes differ from CPU fixtures")
    pairs.require(document.get('failed_empty_cache_inventory',{}).get('qualifies_actual_pair') is False,"selection anchor promoted to actual-pair proof")
    pairs.require(document.get('v2_preregistration_sha256')==V2_PREREGISTRATION_SHA256
        and document.get('v2_native_all8_stages')==[6,10,14] and document.get('cached_stage_count')==18
        and document.get('v1_or_case2_pass_transferred') is False,'fresh v2 execution metadata/scope differs')
    selection=document['failed_empty_cache_inventory']
    actual_failure=selection.get('v1_case0_selection',{})
    pairs.require(actual_failure.get('manifest_sha256')==V1_PAIR_FAILURE_SHA256
        and actual_failure.get('failed_stages')==[14] and actual_failure.get('qualification_passed') is False
        and selection.get('predeclared_v2_native_all8_stages')==[6,10,14],
        'recorded v1 failure selection or predeclared v2 routes differ')
    comparison=dict(document);comparison['status']='prior_native_control_screen_pass'
    replay=_numeric_validate(comparison,expected_sources=expected_sources,raw_bindings=raw_bindings,transformed_bindings=transformed_bindings,coordinate_policy=coordinate_policy,native_policy=native_policy,inventory_replay=inventory_replay)
    return {**replay,"pair":spec.as_dict(),"actual_pair_fixed_prefixes_replayed":True,"pair_CPU_receipt_sha256":PAIR_CPU_RECEIPT_SHA256,"old_case2_numerical_pass_inherited":False,"native_all8_stages":[6,10,14],"cached_stages":18,"v2_preregistration_sha256":V2_PREREGISTRATION_SHA256,"scope":"actual pair fixed FIRST1024 plus stress numerical screen; no posterior/population accuracy claim"}


def build_collector(spec,*,data_proxy,source_reader,selection_reader,writer,validator):
    original=ast.parse(inspect.getsource(numeric.run)).body[0];candidate=copy.deepcopy(original)
    _backend_imports(candidate)
    edits=[]
    for node in ast.walk(candidate):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='derive_seed':
            pairs.require(len(node.args)==2 and isinstance(node.args[0],ast.Constant) and node.args[0].value==20261111,"frozen922 seed call changed")
            kw=next(x for x in node.keywords if x.arg=='case');pairs.require(isinstance(kw.value,ast.Constant) and kw.value.value==2,"frozen922 case seed changed")
            node.args[0]=ast.Attribute(value=ast.Name(id='_PAIR_SPEC',ctx=ast.Load()),attr='context_seed',ctx=ast.Load())
            kw.value=ast.Attribute(value=ast.Name(id='_PAIR_SPEC',ctx=ast.Load()),attr='case',ctx=ast.Load());edits.append(node)
    pairs.require(len(edits)==1,"one explicit pair-seed AST change required")
    reverse=copy.deepcopy(candidate);_backend_imports(reverse,True)
    for node in ast.walk(reverse):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='derive_seed':
            node.args[0]=ast.Constant(20261111);next(x for x in node.keywords if x.arg=='case').value=ast.Constant(2)
    pairs.require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),"undeclared numeric collector AST change")
    namespace=dict(vars(numeric));namespace.update(_PAIR_SPEC=spec,SCHEMA=SCHEMA,data_helper=lambda:data_proxy,source_pins=source_reader,failed_screen_receipt=selection_reader,write_json=writer,validate_screen=validator)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[])),__file__+'::private922_pair_seed_backend_v2','exec'),namespace)
    return namespace['run']


def run(*,spec,banks,prefix_run,checkpoint,codec_inventory,codec_reference_inventory,codec_cpu_receipt,completed_grid_audit,failed_screen,selection_anchor,v1_pair_failure,pair_cpu_receipt,output,ledger,max_seconds=1050):
    pairs.require(type(spec) is pairs.PairSpec and spec==pairs.FIRST_PAIR,"only first lexicographic seed11/case0 GPU screen is authorized")
    pins=source_pins();cpu,cpurow=_cpu(pair_cpu_receipt,spec);data=numeric.data_helper()
    owner=pairs.prefix_parent(spec,completion_receipt=completed_grid_audit)
    if prefix_run is not None:pairs.require(Path(prefix_run).resolve()==Path(owner['manifest_path']).parent,"wrong-pair prefix path")
    prefix_run=str(Path(owner['manifest_path']).parent)
    base_inventory=data.inventory_receipt(codec_inventory,codec_reference_inventory,codec_cpu_receipt)[1]
    replay=_inventory_replay(base_inventory,spec)
    proxy=SimpleNamespace(**{k:v for k,v in vars(data).items() if not k.startswith('__')})
    def actual_inputs(banks_arg,parent_arg,_inventory,*,runtime_audit=None):
        return pairs.input_bindings(spec,banks=banks_arg,completion_receipt=completed_grid_audit,training_inventory=codec_inventory,pair_inventory=cpu,old_parent=parent_arg,runtime_audit=runtime_audit)
    def inventory_reader(*args):
        original,observed=data.inventory_receipt(*args);pairs.require(observed==base_inventory,"fixed training inventories changed")
        _cpu(pair_cpu_receipt,spec)
        return original,replay
    proxy.input_bindings=actual_inputs;proxy.inventory_receipt=inventory_reader
    def selection_reader(path,_raw,_transformed):return _selection(path,selection_anchor,v1_pair_failure)
    def decorated(report):
        result=dict(report);result.update(pair=spec.as_dict(),pair_configuration=spec.metadata(),pair_CPU_receipt_sha256=PAIR_CPU_RECEIPT_SHA256,fixed_panel_preregistration_sha256=pairs.PREREGISTRATION_SHA256,no_other_pair_numerical_qualification_inherited=True,v2_preregistration_sha256=V2_PREREGISTRATION_SHA256,v2_native_all8_stages=[6,10,14],cached_stage_count=18,v1_or_case2_pass_transferred=False)
        if result.get('status')=='prior_native_control_screen_pass':result['status']=PASS_STATUS
        elif result.get('status')=='prior_native_control_structural_or_hard_failure':result['status']='prior_coordinate_pair_structural_or_hard_failure'
        return result
    def writer(path,document):write_json(path,decorated(document))
    def validator(document,**kwargs):return validate_screen(decorated(document),spec=spec,pair_cpu_receipt=cpu,**kwargs)
    collector=build_collector(spec,data_proxy=proxy,source_reader=source_pins,selection_reader=selection_reader,writer=writer,validator=validator)
    report=collector(banks=banks,prefix_run=prefix_run,checkpoint=checkpoint,codec_inventory=codec_inventory,codec_reference_inventory=codec_reference_inventory,codec_cpu_receipt=codec_cpu_receipt,completed_grid_audit=completed_grid_audit,failed_screen=failed_screen,output=output,ledger=ledger,max_seconds=max_seconds)
    pairs.require(source_pins()==pins,"new pair qualifier source closure changed")
    return decorated(report)


def qualification_receipt(spec,artifact_path,*,banks,completion_receipt,training_inventory,reference_training_inventory,combined_training_inventory,pair_cpu_receipt,raw_bindings,model_inputs,prefix_run=None,failed_screen=None):
    from posteriorforge import execution_prior_native_v2 as native
    from posteriorforge.science.analytical_benchmark_v2 import array_descriptor
    pins=source_pins();cpu,row=_cpu(pair_cpu_receipt,spec)
    theta,stats,query,_,raw,transformed,custody=pairs.input_bindings(spec,banks=banks,completion_receipt=completion_receipt,training_inventory=training_inventory,pair_inventory=cpu,old_parent=prefix_run)
    pairs.require(raw==raw_bindings and model_inputs=={'context_theta':array_descriptor(theta),'context_obs':array_descriptor(stats),'query':array_descriptor(query)},"actual producer pair/raw/model inputs differ from screen")
    data=numeric.data_helper();base=data.inventory_receipt(training_inventory,reference_training_inventory,combined_training_inventory)[1]
    d=json.loads(Path(artifact_path).read_bytes())
    receipt=validate_screen(d,spec=spec,expected_sources=pins,raw_bindings=raw,transformed_bindings=transformed,coordinate_policy=native.cached.policy_metadata(),native_policy=native.policy_metadata(),inventory_replay=_inventory_replay(base,spec),pair_cpu_receipt=cpu)
    if failed_screen is not None:pairs.require(pairs.sha(failed_screen)==numeric.FAILED_SCREEN_SHA256,"immutable selection failure artifact changed")
    return d,{'screen_sha256':pairs.sha(artifact_path),'replay':receipt,'pair':spec.as_dict(),'raw_scientific_bindings':raw,'transformed_bindings':transformed,'fixed_prefix_custody':custody,'pair_CPU_receipt_sha256':PAIR_CPU_RECEIPT_SHA256,'scope':'separately performed actual-pair finite-prefix screen; request only CPU-replays artifact','FP16_activated':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--context-seed',type=int,required=True);p.add_argument('--case',type=int,required=True)
    for key in ['banks','prefix-run','checkpoint','codec-inventory','codec-reference-inventory','codec-cpu-receipt','completed-grid-audit','failed-screen','selection-anchor','v1-pair-failure','pair-cpu-receipt','output','ledger']:p.add_argument('--'+key,required=True)
    p.add_argument('--max-seconds',type=int,default=1050);args=vars(p.parse_args());spec=pairs.PairSpec(args.pop('context_seed'),args.pop('case'));d=run(spec=spec,**args)
    print(json.dumps({k:d.get(k) for k in ['schema','status','pair','matching_FP32_execution_passed_all21','FP32_failed_stages']},sort_keys=True))


if __name__=='__main__':main()
