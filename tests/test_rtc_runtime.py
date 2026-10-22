import unittest
import numpy as np
from inference.rtc.runtime import PreparedSampler, audit_normalization


def flatten(value):
    if isinstance(value,dict):
        pairs=[(key,flatten(item)) for key,item in sorted(value.items())]
        return [x for _,(leaves,_) in pairs for x in leaves],tuple((k,t) for k,(_,t) in pairs)
    if isinstance(value,tuple):
        pairs=[flatten(item) for item in value]
        return [x for leaves,_ in pairs for x in leaves],tuple(t for _,t in pairs)
    return [value],'leaf'


class RuntimeTests(unittest.TestCase):
    def test_dynamic_values_share_prepared_signature_but_new_shapes_do_not(self):
        calls=[]
        sampler=PreparedSampler(lambda *a,**kw:calls.append(1) or a[0],flatten,lambda x:None)
        sampler(np.zeros((1,20,32),np.float32),cap=1000.)
        sampler.sealed=True
        sampler(np.ones((1,20,32),np.float32),cap=1000.)
        self.assertTrue(sampler.last['prepared_before_call'])
        for args,kw in [((np.zeros((1,14,32),np.float32),),{'cap':1000.}),
                        ((np.zeros((1,20,32),np.float64),),{'cap':1000.}),
                        ((np.zeros((1,20,32),np.float32),),{'cap':5.}),
                        ((np.zeros((1,20,32),np.float32),),{'cap':1000.,'noise':np.zeros((1,20,32),np.float32)})]:
            with self.assertRaisesRegex(RuntimeError,'not warmed'):sampler(*args,**kw)
        self.assertEqual(len(calls),2)

    def test_offline_experiment_does_not_expand_online_signatures(self):
        sampler=PreparedSampler(lambda x,**kw:x,flatten,lambda x:None)
        x=np.zeros((1,20,32),np.float32);sampler(x);sampler.sealed=True
        with self.assertRaisesRegex(ValueError,'test'):
            with sampler.offline_experiment():
                sampler(x,cap=100.)
                self.assertTrue(sampler.last['offline_diagnostic'])
                raise ValueError('test')
        self.assertFalse(sampler.offline)
        with self.assertRaises(RuntimeError):sampler(x,cap=100.)

    def test_roundtrip_audit_catches_clipping(self):
        def normalize(d):return {k:(v-.1)/.25 for k,v in d.items()}
        def unnormalize(d):return {k:v*.25+.1 for k,v in d.items()}
        self.assertEqual(audit_normalization(normalize,unnormalize)['valid_dimensions'],7)
        with self.assertRaisesRegex(ValueError,'roundtrip failed'):
            audit_normalization(normalize,lambda d:{k:np.clip(v,-1,1)*.25+.1 for k,v in d.items()})


if __name__=='__main__':unittest.main()
