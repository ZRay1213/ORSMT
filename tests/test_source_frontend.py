import copy
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from orsmt.language_frontend import ChatJSONClient, MathematicalTranscriber, SourceGrounder, digest
from orsmt.source_check import SourceChecker
from orsmt.translator import ModelTranslator
from orsmt.workflow import ORSMTWorkflow
from orsmt.milp_codegen import model_to_dict
from orsmt.ir import ResultRecord
from orsmt.repair_engine import ProgramExecution

SOURCE='Choose an integer x between 0 and 10, with x at least 2. Minimize x.'
def formulation(bound=2):
    return {'variables':[{'name':'x','type':'integer','lb':'0','ub':'10','observable':True}],
            'constraints':[{'name':'minimum','lhs':[{'var':'x','coef':'1'}],'constant':'0','op':'>=','rhs':str(bound)}],
            'objective':{'sense':'min','lhs':[{'var':'x','coef':'1'}],'constant':'0'}}
def proposal(complete=True):
    return dict(formulation=formulation(),requirements=[{'id':kind,'kind':kind,'targets':[target],
        'source_spans':[{'document':'source','start':0,'end':len(SOURCE),'quote':SOURCE}]}
        for kind,target in [('variable','x'),('constraint','minimum'),('objective','objective')]],
        ambiguities=[],missing_information=[],complete=complete)
class Grounder:
    def __init__(self,p=None,approved=True):self.p=p or proposal();self.approved=approved;self.inputs=[]
    def propose(self,s,d):self.inputs.append((s,d));return copy.deepcopy(self.p)
    def review(self,s,d,p,h):
        self.inputs.append((s,d,p,h))
        return dict(reviewed_sha256=h,approved=self.approved,complete=p['complete'],issues=[],rejected_requirement_ids=[],uncovered_source_spans=[])
    def explain(self,s,r):return {'source_quotes':[SOURCE],'explanations':['test explanation']}

class SourceTests(unittest.TestCase):
    def setup_checker(self,p=None,approved=True):
        g=Grounder(p,approved);c=SourceChecker(g);return c,c.prepare(SOURCE)
    def test_complete_certification_and_discrepancies(self):
        c,p=self.setup_checker()
        self.assertEqual(p['status'],'REVIEW_APPROVED',p)
        self.assertEqual(c.check(p,formulation())['verdict'],'CERTIFIED')
        for b in (1,3):
            r=c.check(p,formulation(b));self.assertEqual(r['verdict'],'REFUTED',r)
            self.assertFalse(r['natural_language_certified'])
            self.assertTrue(any(x['witness'] for x in r['records']))
        bad=formulation();bad['variables'][0]['type']='continuous'
        self.assertEqual(c.check(p,bad)['verdict'],'REFUTED')
        bad=formulation();bad['objective']['constant']='1'
        self.assertEqual(c.check(p,bad)['verdict'],'REFUTED')
    def test_partial_never_full_pass(self):
        c,p=self.setup_checker(proposal(False))
        self.assertEqual(c.check(p,formulation())['verdict'],'UNDECIDED')
        self.assertEqual(c.check(p,formulation(1))['verdict'],'REFUTED')
    def test_rejected_ambiguous_and_bad_provenance(self):
        c,p=self.setup_checker(approved=False);self.assertEqual(c.check(p,formulation())['verdict'],'UNDECIDED')
        q=proposal();q['ambiguities']=['unclear domain'];c,p=self.setup_checker(q)
        self.assertEqual(p['status'],'REVIEW_NOT_APPROVED')
        q=proposal();q['requirements'][0]['source_spans'][0]['quote']='fake';c,p=self.setup_checker(q)
        self.assertEqual(p['status'],'UNDECIDED')
        q=proposal();del q['formulation']['variables'][0]['type'];c,p=self.setup_checker(q)
        self.assertEqual(p['status'],'UNDECIDED')
    def test_tamper_and_correspondence(self):
        c,p=self.setup_checker();bad=copy.deepcopy(p);bad['model']['constraints'][0]['rhs']='3'
        with self.assertRaisesRegex(ValueError,'modified'):c.check(bad,formulation())
        renamed=formulation();renamed['variables'][0]['name']='z'
        renamed['constraints'][0]['lhs'][0]['var']='z';renamed['objective']['lhs'][0]['var']='z'
        self.assertEqual(c.check(p,renamed)['verdict'],'UNDECIDED')
        self.assertEqual(c.check(p,renamed,variable_map={'x':'z'})['verdict'],'CERTIFIED')
    def test_workflow_both_wrong_and_repair(self):
        c,p=self.setup_checker();w=ORSMTWorkflow(source_checker=c)
        out=w.check(case_id='bothwrong',written=formulation(3),program=formulation(3),source_product=p)
        self.assertEqual(out['diagnosis']['implementation']['status'],'CERTIFIED')
        self.assertTrue(all(v['verdict']=='REFUTED' for v in out['source_checks'].values()))
        def execute(code):
            b=3 if code=='original' else 2
            return ProgramExecution(ModelTranslator().translate(formulation(b)).model,ResultRecord({'x':b},b,'optimal'))
        out=w.repair_from_source(source_product=p,case_id='bothwrong',written=formulation(3),code='original',execute=execute)
        self.assertTrue(out['accepted'],out)
        self.assertEqual(out['source_recheck']['verdict'],'CERTIFIED')
        self.assertFalse(out['natural_language_verified'])
        self.assertEqual(len(c.grounder.inputs),2)
        self.assertNotIn('candidate',str(c.grounder.inputs))
    def test_review_binding(self):
        class Wrong(Grounder):
            def review(self,*args):
                r=super().review(*args);r['reviewed_sha256']='wrong';return r
        c=SourceChecker(Wrong());self.assertEqual(c.prepare(SOURCE)['status'],'UNDECIDED')

class TransportTests(unittest.TestCase):
    def setUp(self):
        owner=self;self.requests=[];self.response_model='fixture';self.finish='stop';self.reply='{}'
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                response={'model':owner.response_model,'choices':[{'finish_reason':owner.finish,'message':{'content':owner.reply}}],
                          'usage':{'prompt_tokens':10,'completion_tokens':20}}
                payload=json.dumps(response).encode();self.send_response(200);self.end_headers();self.wfile.write(payload)
            def log_message(self,*args):pass
        self.server=HTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.client=ChatJSONClient('http://127.0.0.1:'+str(self.server.server_port)+'/v1','fixture',timeout=2)
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def test_real_http_transcription_path(self):
        self.reply=json.dumps(dict(status='ok',formulation=formulation(),assumptions=[],source_spans=['x >= 2']))
        t=ModelTranslator.from_client(self.client).translate('min x; integer 0 <= x <= 10; x >= 2',format='text')
        self.assertEqual(t.model.constraints[0].rhs,2);self.assertEqual(t.metadata()['translation_llm_calls'],1)
        self.assertEqual(t.details['usage']['completion_tokens'],20)
        self.assertEqual(len(self.requests),1);self.assertFalse(t.details['translation_equivalence_certified'])
    def test_identity_truncation_and_duplicate_fail_closed(self):
        self.response_model='other'
        with self.assertRaisesRegex(ValueError,'identity'):self.client.complete('s',{},stage='test')
        self.response_model='fixture';self.finish='length'
        with self.assertRaisesRegex(ValueError,'incomplete'):self.client.complete('s',{},stage='test')
        self.finish='stop';self.reply='{"x":1,"x":2}'
        with self.assertRaisesRegex(ValueError,'duplicate'):self.client.complete('s',{},stage='test')
        self.assertEqual(len(self.client.calls),3);self.assertTrue(all(x['status']=='failed' for x in self.client.calls))
    def test_no_hidden_free_text_calls(self):
        t=ModelTranslator.from_client(self.client)
        with self.assertRaises(ValueError):t.translate('a model',format='text',allow_llm=False)
        self.assertEqual(self.requests,[])
    def test_transcriber_rejects_missing_domains(self):
        f=formulation();del f['variables'][0]['type']
        self.reply=json.dumps(dict(status='ok',formulation=f,assumptions=[],source_spans=['x']))
        with self.assertRaises(ValueError):ModelTranslator.from_client(self.client).translate('x',format='latex')

    def test_grounder_metadata_and_numeric_ids(self):
        p=proposal()
        p['formulation']['variables'][0]['provenance']={'quote':SOURCE}
        for i,r in enumerate(p['requirements']):
            r['id']=i
            for span in r['source_spans']:span.pop('start');span.pop('end')
        self.reply=json.dumps(p)
        out=SourceGrounder(self.client,self.client).propose(SOURCE,'')
        from orsmt.source_check import validate_proposal
        validate_proposal(SOURCE,'',out)
        self.assertEqual(out['requirements'][0]['id'],'0')
        self.assertEqual(out['requirements'][0]['source_spans'][0]['offset_basis'],'unique_exact_quote')

    def test_grounder_ambiguous_quote_fails(self):
        p=proposal();p['requirements'][0]['source_spans'][0]['quote']='x'
        self.reply=json.dumps(p)
        with self.assertRaisesRegex(ValueError,'unambiguous'):SourceGrounder(self.client,self.client).propose(SOURCE,'')

if __name__=='__main__':unittest.main()
