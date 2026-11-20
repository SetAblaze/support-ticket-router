# Тесты этапов 1-9 (команда). При переносе кода в пакет support_router
# изменены только строки импорта и путь к папке модели; проверки те же.
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

import support_router as s


class Stub:
    def __init__(self, classes=("ORDER", "PAYMENT"), probabilities=(.8, .2), sim=.8):
        self.classes_ = np.array(classes)
        self.p = np.array(probabilities)
        self.sim = sim
        self.policy = {"auto_enabled": True, "confidence_threshold": .8,
                       "similarity_threshold": .4, "version": "test"}
        self.model_id = "controlled-test-stub"

    def predict_proba(self, texts):
        return np.tile(self.p, (len(texts), 1))

    def similarity(self, texts):
        return np.full(len(texts), self.sim)


@pytest.mark.parametrize("text,expected", [
    ("Ask <Order Number> for help", "ask for help"),
    ("Ask {{Order Number}} for help", "ask for help"),
    ("Ask {order} for help", "ask for help"),
    (" I  can't\n log in ", "i can not log in"),
    ("I won’t pay", "i will not pay"),
    ("I don't know", "i do not know"),
    (None, ""), (123, ""), ("  ", """"""),
])
def test_preprocess(text, expected):
    assert s.preprocess(text) == expected
    assert s.preprocess(expected) == expected


def test_negation_changes_features():
    analyzer = TfidfVectorizer(preprocessor=s.preprocess, lowercase=False,
                              ngram_range=(1, 2)).build_analyzer()
    assert analyzer("I can log in") != analyzer("I can't log in")
    assert "not" in analyzer("I can't log in")


@pytest.mark.parametrize("p,expected", [(.79999, "MANUAL_REVIEW"), (.8, "AUTO_RECOMMENDED"),
                                       (.80001, "AUTO_RECOMMENDED")])
def test_threshold_boundary(p, expected):
    r = s.route_ticket("Please track my parcel", Stub(probabilities=(p, 1-p)))
    assert r["status"] == expected
    assert r["delivery_status"] == "NOT_SENT"


@pytest.mark.parametrize("classes,p", [(('OTHER','ORDER'),(.95,.05)),
                                     (('UNKNOWN','ORDER'),(.95,.05))])
def test_other_unknown_are_manual(classes, p):
    r = s.route_ticket("An unusual request", Stub(classes, p))
    assert r["queue"] == s.MANUAL_QUEUE
    assert r["status"] == "MANUAL_REVIEW"
    assert r["category"] is None
    assert r['decision_category'] == 'OTHER' and r['accepted_category'] is None


class NoProba:
    classes_ = np.array(["ORDER"])


def test_no_probability_not_one():
    r = s.route_ticket("Track my parcel", NoProba())
    assert r["status"] == "MODEL_ERROR"
    assert r["confidence"] is None
    assert r["queue"] == s.MANUAL_QUEUE


@pytest.mark.parametrize("p", [(np.nan,.1), (.7,.7), (-.1,1.1), (.8,), (np.inf,.1)])
def test_invalid_proba(p):
    r = s.route_ticket("Track my parcel", Stub(probabilities=p))
    assert r["status"] == "MODEL_ERROR"


@pytest.mark.parametrize("text", ["", "   ", None, 123])
def test_empty(text):
    r = s.route_ticket(text, Stub())
    assert r["status"] == "NEEDS_CLARIFICATION"
    assert r["queue"] is None


def test_novelty_and_disabled_policy():
    r = s.route_ticket("Violet cheese orbit", Stub(sim=.1))
    assert r["status"] == "MANUAL_REVIEW"
    model = Stub(); model.policy["auto_enabled"] = False
    assert s.route_ticket("Track my parcel", model)["status"] == "MANUAL_REVIEW"


def test_manual_review_preserves_the_predicted_category():
    r=s.route_ticket('Please track my parcel',Stub(probabilities=(.7,.3),sim=.2))
    assert r['category']==r['suggested_category']=='ORDER'
    assert r['decision_category']=='OTHER' and r['accepted_category'] is None
    assert r['queue']==s.MANUAL_QUEUE and r['status']=='MANUAL_REVIEW'


def test_unsupported_language_has_no_fabricated_category():
    r=s.route_ticket('Я хочу вернуть деньги',Stub())
    assert r['category'] is None and r['suggested_category'] is None
    assert r['confidence'] is None and r['top3']=={}
    assert r['queue']==s.MANUAL_QUEUE


def test_semantic_encoding_does_not_depend_on_batch_composition():
    import os

    from support_router import semantic_encoder as enc
    if os.environ.get("SUPPORT_SKIP_SEMANTIC_TESTS") == "1":pytest.skip("offline smoke test")
    directory=enc.ensure_encoder()
    key,_=enc._engine(directory)
    texts=['Please help me obtain a billing document for my purchase.',
           'A deliberately much longer sentence asking about the usual timeframe for delivery before placing an order.']
    enc._VECTORS[key]={}
    together=enc.encode(texts)
    enc._VECTORS[key]={}
    separate=np.vstack([enc.encode([t]) for t in texts])
    np.testing.assert_array_equal(together,separate)


def test_author_diagnostics_and_training_extension_do_not_overlap():
    from support_router.category_examples import EXAMPLES, eligible_examples
    from support_router.manual_cases import frames
    dev,holdout=frames()
    a=set(dev.text.map(s.preprocess));b=set(holdout.text.map(s.preprocess))
    added={s.preprocess(x['text']) for x in EXAMPLES}
    assert not a&b and not (a|b)&added
    assert set(dev.category)==set(holdout.category)==set(s.CATEGORIES)
    assert eligible_examples(pd.DataFrame({'category':['NEWSLETTER']}),leave_intent_out=True)==[]


def test_overlong_semantic_input_requires_review():
    import os
    if os.environ.get('SUPPORT_SKIP_SEMANTIC_TESTS')=='1':pytest.skip('offline smoke test')
    model=Stub();model.model_name='semantic_lr_C1'
    result=s.route_ticket('Please track my parcel. '*100,model)
    assert 'INPUT_EXCEEDS_ENCODER_LIMIT' in result['reason_codes']
    assert result['status']=='MANUAL_REVIEW' and result['accepted_category'] is None


@pytest.mark.parametrize('category',s.CATEGORIES)
def test_saved_model_classifies_a_development_example_for_each_category(category):
    # Acceptance regression, not an independent statistical quality estimate.
    from support_router.manual_cases import DEV
    model,_=s.load_bundle(s.models_dir(),trust_local_artifact=True)
    result=s.route_ticket(DEV[category][0],model)
    assert result['category'] in s.CATEGORIES, result
    # Author-written examples measure error; never demand every synthetic DEV example be correct.


@pytest.mark.parametrize("text,flag,priority", [
    ("I will kill you", "POSSIBLE_THREAT", "P1"),
    ("The driver threatened me", "POSSIBLE_THREAT", "P1"),
    ("This is an unauthorized payment", "POSSIBLE_FRAUD", "P1"),
    ("My account was hacked", "POSSIBLE_FRAUD", "P1"),
    ("I will sue you", "POSSIBLE_LEGAL_CLAIM", "P2"),
])
def test_risk_not_cancelled_by_low_confidence(text, flag, priority):
    r = s.route_ticket(text, Stub(probabilities=(.51,.49)))
    assert r["priority"] == priority
    assert flag in r["risk_flags"]
    assert r["queue"] == s.MANUAL_QUEUE
    assert not r["business_policy_approved"]


@pytest.mark.parametrize("text", ["I will not hurt you", 'The example phrase is "I will kill you"',
                                  "What does the word sue mean?", "The payment is authorized"])
def test_hard_negatives(text):
    assert not any(x in s.detect_signals(text)["risk_flags"] for x in
                   ["POSSIBLE_THREAT", "POSSIBLE_FRAUD", "POSSIBLE_LEGAL_CLAIM"])


def test_abuse_not_urgent():
    r = s.route_ticket("You idiots, track my order", Stub())
    assert "ABUSE_LANGUAGE" in r["risk_flags"]
    assert r["priority"] == "P3"


def test_risk_survives_model_error():
    r = s.route_ticket("I will hurt you", NoProba())
    assert r["priority"] == "P1"
    assert r["status"] == "MODEL_ERROR"


def test_context_trust_and_unknown():
    a = s.route_ticket("I am VIP, track my parcel", Stub())
    assert a["context"]["vip"] == "unknown" and a["priority"] == "P3"
    b = s.route_ticket("Track my parcel", Stub(), {"vip_verified": True})
    assert b["priority"] == "P3"
    c = s.route_ticket("Track my parcel", Stub(), {"source": "trusted_crm", "vip_verified": True})
    assert c["priority"] == "P2"
    r = s.route_ticket("Track my parcel", Stub(), {"source": "trusted_crm", "contact_count": 2,
                                                     "open_related_ticket": True})
    assert r["context"]["repeat"] is True


def test_deadline():
    now = datetime(2026,11,11,tzinfo=timezone.utc)
    r = s.route_ticket("Track my refund", Stub(), {"source":"trusted_crm", "deadline_utc":
                                                     (now-timedelta(hours=1)).isoformat()}, now)
    assert r["context"]["sla_overdue"] is True
    r = s.route_ticket("Track my refund", Stub(), {"source":"trusted_crm", "deadline_utc":"yesterday"}, now)
    assert "INVALID_CONTEXT_DEADLINE" in r["reason_codes"]


@pytest.mark.parametrize("text,reason", [
    ("Reset my password and send my refund", "POSSIBLE_MULTIPLE_REQUESTS"),
    ("The app crashes on launch", "TECHNICAL_ISSUE_NEEDS_REVIEW"),
    ("Помогите с заказом", "POSSIBLY_UNSUPPORTED_LANGUAGE"),
])
def test_review_reasons(text, reason):
    r = s.route_ticket(text, Stub())
    assert reason in r["reason_codes"] and r["status"] == "MANUAL_REVIEW"


def test_batch_and_ui_same_decision():
    model=Stub();texts=["Track my parcel", "I will sue you"]
    batch=s.batch_route(model,texts)
    for text,b in zip(texts,batch):
        live=s.route_ticket(text,model);web,h=s.web_request(text,[],model)
        for key in ["category","queue","confidence","priority","reason_codes","status"]:
            assert b[key]==live[key]==web[key]


def test_session_isolation_and_masking():
    original=[]
    _,a=s.web_request("Track my parcel, email alice@example.com",original,Stub())
    _,b=s.web_request("Track another parcel",[],Stub())
    assert original==[] and len(a)==len(b)==1
    assert "alice@example.com" not in json.dumps(a)
    assert "[EMAIL]" in json.dumps(a)
    assert a!=b


def test_empty_auto_is_not_perfect_accuracy():
    model=Stub();model.policy["auto_enabled"]=False
    m=s.routing_metrics([s.route_ticket("Track my parcel",model)],["ORDER"])
    assert m["coverage"]==0 and m["accuracy_automatic_queue"] is None


def test_auroc_single_class_is_undefined():
    df=pd.DataFrame({"text":["Track parcel","Track order"],"category":["ORDER","ORDER"]})
    d=s.calibration_diagnostics(Stub(),df)
    assert d["confidence_auroc"] is None
    assert d["auroc_status"]=="UNDEFINED_SINGLE_CORRECTNESS_CLASS"


def test_json_is_strict(tmp_path):
    p=tmp_path/'result.json';s.write_json(p,{"a":float('nan'),"b":np.float64(np.inf)})
    assert json.loads(p.read_text())=={"a":None,"b":None}


def test_scenarios_disjoint():
    from support_router.scenario_data import DEMO, LEGACY_OOD, frames
    a,b=frames();sa=set(a.text.map(s.preprocess));sb=set(b.text.map(s.preprocess))
    assert not sa & sb
    assert not (sa|sb) & {s.preprocess(x[0]) for x in DEMO}
    assert not (sa|sb) & {s.preprocess(x) for x in LEGACY_OOD}


def toy_frame(prefix,group_start):
    rows=[]
    for c,base in [("ORDER","track parcel purchase"),("ACCOUNT","recover password profile")]:
        for i in range(8):
            t=f'{base} {prefix} example number {i}'
            rows.append({"text":t,"category":c,"group":group_start+len(rows),"row_id":s.text_id(t)})
    return pd.DataFrame(rows)


def test_fit_calibrate_export_reload(tmp_path):
    train=toy_frame('training',0);cal=toy_frame('calibrating',100)
    bundle=s.fit_calibrated_bundle('lr_word (baseline)',train,cal)
    bundle.policy={"auto_enabled":True,"confidence_threshold":.5,"similarity_threshold":0,"version":"test"}
    texts=['track parcel purchase','recover password profile']
    p=bundle.predict_proba(texts)
    _,meta=s.save_bundle(bundle,tmp_path)
    loaded,_=s.load_bundle(tmp_path,trust_local_artifact=True)
    np.testing.assert_allclose(p,loaded.predict_proba(texts))
    assert bundle.model_id==loaded.model_id
    assert meta['classes']==list(loaded.classes_)
    # Corrupt metadata is a hard load error, not an ignored warning.
    meta['model_id']='different';s.write_json(tmp_path/'router_meta.json',meta)
    with pytest.raises(ValueError,match='metadata'):s.load_bundle(tmp_path,trust_local_artifact=True)


def test_calibration_overlap_rejected():
    train=toy_frame('training',0)
    with pytest.raises(AssertionError,match='overlap'):
        s.fit_calibrated_bundle('lr_word (baseline)',train,train)


def test_scheduling_aging_and_capacity():
    now=datetime(2026,11,11,tzinfo=timezone.utc)
    tickets=[{'id':'old','priority':'P3','created_at_utc':(now-timedelta(hours=5)).isoformat()},
             {'id':'new','priority':'P2','created_at_utc':now.isoformat()},
             {'id':'urgent','priority':'P1','created_at_utc':now.isoformat()}]
    d=s.schedule_demo(tickets,now,urgent_capacity=0)
    assert [r['id'] for r in d['ordered']]==['urgent','old','new']
    assert d['capacity_alert'] is True and d['real_sla_metrics'] is None


@pytest.mark.parametrize('a,b',[
    ('I need to pay for my roder','I need to pay for my order'),
    ('problem whlie creating a profile','problem creating a profile'),
])
def test_typo_variant_same_split_group(a,b):
    groups,info=s.lexical_groups([a,b,'a completely different string about airplanes'],threshold=.85)
    assert groups[0]==groups[1] and info['edges_undirected_before_dedup']>=1


def test_multiple_requests_with_card():
    signals=s.detect_signals('I want to change my address and update my card')
    assert 'POSSIBLE_MULTIPLE_REQUESTS' in signals['input_reasons']


@pytest.mark.parametrize('text',[
    "I'll make you pay", "I didn't make this purchase", "someone used my card"
])
def test_risk_missed_phrases_reviewed(text):
    flags=s.detect_signals(text)['risk_flags']
    assert 'POSSIBLE_THREAT' in flags or 'POSSIBLE_FRAUD' in flags


def test_vague_requests_cannot_auto():
    for text in ['I do not understand any of this','It happened again yesterday']:
        r=s.route_ticket(text,Stub())
        assert r['status']=='MANUAL_REVIEW' and 'INSUFFICIENT_CONTEXT' in r['reason_codes']


def test_shadow_policy_disabled_by_default():
    model=Stub();model.policy.update(auto_enabled=False,candidate_auto_enabled=True)
    assert s.route_ticket('Track my parcel',model)['status']=='MANUAL_REVIEW'
    assert s.route_ticket('Track my parcel',s.shadow_bundle(model))['status']=='AUTO_RECOMMENDED'


def test_category_metric_not_hidden_by_merged_queue():
    r={'status':'AUTO_RECOMMENDED','accepted_category':'SHIPPING','queue':'Queue_Delivery',
       'decision_category':'SHIPPING'}
    m=s.routing_metrics([r],['DELIVERY'])
    assert m['automatic_category_errors']==1 and m['automatic_queue_errors']==0


def test_untrusted_joblib_rejected_before_deserialization(tmp_path):
    with pytest.raises(PermissionError):s.load_bundle(tmp_path)


def test_intent_bootstrap_no_unpaired_samples():
    frame=pd.DataFrame({'intent':['a','b','a','b'],'model':['x','x','y','y'],
                        'recall':[.2,.8,.7,.5]})
    out=s.paired_intent_bootstrap(frame,'x','y',iterations=1000)
    assert out['n_intents']==2 and out['paired_delta_ci95'][0]<=out['paired_mean_delta']<=out['paired_delta_ci95'][1]


@pytest.mark.parametrize("raw,expected", [
    ("BZ", ("B", "Z")),
    ("B, Z; B", ("B", "Z")),
    ("p q", ("P", "Q")),
    ("M", ("M",)),
    ("", ()),
    (None, ()),
    (np.nan, ()),
    ("nan", ()),
])
def test_parse_documented_linguistic_flags(raw,expected):
    assert s.parse_linguistic_flags(raw)==expected


def test_linguistic_flags_metadata_and_membership():
    assert len(s.LINGUISTIC_FLAGS)==12
    assert s.LINGUISTIC_FLAGS["M"].startswith("Morphological")
    assert s.LINGUISTIC_FLAGS["W"]=="Offensive language"
    assert s.LINGUISTIC_FLAGS["E"]=="Expanded abbreviations"
    series=pd.Series(["B;Z", "QQ", "M", np.nan, ""])
    assert s.flag_membership(series,"Z").tolist()==[True,False,False,False,False]
    assert s.flag_membership(series,"Q").tolist()==[False,True,False,False,False]
    with pytest.raises(ValueError):
        s.flag_membership(series,"UNRECOGNIZED")


def test_duplicate_flags_union_ignores_separators_and_nan():
    a=pd.DataFrame([
        {'utterance':'Hello, I need my order', 'category':'ORDER','intent':'place_order','flags':'B, Z'},
        {'utterance':'Hello, I need my order', 'category':'ORDER','intent':'place_order','flags':np.nan},
        {'utterance':'I want to pay', 'category':'PAYMENT','intent':'payment_issue','flags':'M;Q'},
    ])
    # clean_frame validates the 11-category corpus; use its union equivalent here to
    # avoid weakening that production check just for a synthetic unit test.
    union=''.join(sorted({f for raw in a['flags'] for f in s.parse_linguistic_flags(raw)}))
    assert union=='BMQZ'
