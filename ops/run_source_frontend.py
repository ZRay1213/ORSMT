"""Auditable source/frontend integration run. Not a population benchmark score."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'ops'))
from orsmt.language_frontend import ChatJSONClient, SourceGrounder, dumps, digest
from orsmt.source_check import SourceChecker
from orsmt.translator import ModelTranslator
from orsmt.workflow import ORSMTWorkflow
from orsmt.milp_codegen import model_to_dict
from run_routed_milp_repair import native_executor


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--endpoint',required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--review-endpoint');ap.add_argument('--review-model')
    ap.add_argument('--reuse-product',type=Path);ap.add_argument('--mapping',type=Path)
    ap.add_argument('--repair',action='store_true');ap.add_argument('--explain',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    def save(name,value):(args.out/name).write_text(dumps(value)+'\n',encoding='utf-8')
    source=json.loads(args.input.read_text());save('input.json',source)
    calls=[]
    def record(value):
        calls.append(value)
        with (args.out/'llm_calls.jsonl').open('a') as f:f.write(dumps(value)+'\n')
    proposer=ChatJSONClient(args.endpoint,args.model,max_tokens=3072,timeout=180,recorder=record)
    reviewer=ChatJSONClient(args.review_endpoint or args.endpoint,args.review_model or args.model,max_tokens=3072,timeout=180,recorder=record)
    translator=ModelTranslator.from_client(proposer)
    checker=SourceChecker(SourceGrounder(proposer,reviewer),translator=translator)
    workflow=ORSMTWorkflow(translator=translator,source_checker=checker)
    summary={'case':source['id'],'role':'integration_case_not_population_accuracy','natural_language_certified':False}
    try:
        # No candidate enters either source grounding request.
        product=json.loads(args.reuse_product.read_text()) if args.reuse_product else workflow.prepare_source(source['question'],data=source.get('data',''))
        checker.validate_product(product)
        if product['source_sha256']!=digest({'source':source['question'],'data':source.get('data','')}):raise ValueError('reused source differs')
        save('source_product.json',product);summary['source_status']=product['status'];summary['source_complete']=product['complete']
        print('Source:',product['status'],'complete:',product['complete'],flush=True)
        written=translator.translate(source['formulation'],format=source.get('formulation_format','auto'))
        save('written_translation.json',{'metadata':written.metadata(),'formulation':model_to_dict(written.model)})
        execute=native_executor();program=execute(source['code'])
        save('program_capture.json',model_to_dict(program.model))
        mapping=json.loads(args.mapping.read_text()) if args.mapping else None
        report=workflow.check(case_id=source['id'],written=written.model,program=program.model,result=program.result,
                              source_product=product,source_variable_maps=mapping)
        save('checks.json',report)
        summary['implementation']=report['diagnosis']['implementation']['status']
        summary['source_checks']={k:v['verdict'] for k,v in report['source_checks'].items()}
        if args.explain:
            explanation=checker.explain(product,report['source_checks']['written']);save('explanation.json',explanation)
        if args.repair:
            if mapping:raise ValueError('repair requires canonical source/artifact names; explicit mapped checks are supported')
            repair=workflow.repair_from_source(source_product=product,case_id=source['id'],written=written.model,code=source['code'],execute=execute)
            save('repair.json',repair);summary['repair']={k:repair[k] for k in ('status','accepted','acceptance_scope','initial_result','result')}
        summary['status']='completed'
    except Exception as exc:
        summary.update(status='unresolved',error_type=type(exc).__name__,reason=str(exc))
    summary['new_llm_calls']=len(calls)
    summary['usage']={k:sum((c.get('usage') or {}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    save('summary.json',summary)
    save('MANIFEST.json',{p.name:digest(p.read_text()) for p in args.out.iterdir() if p.is_file()})
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if summary['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
