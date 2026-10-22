import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('rtc_compact_codec',Path(__file__).parents[1]/'inference/rtc/codec.py')
codec=importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


class CompactInputsTests(unittest.TestCase):
    def inputs(self):
        mask=np.zeros(200,bool);mask[[0,2]]=True
        return {'image_mask':{'right_wrist_0_rgb':np.False_},
                'tokenized_prompt':np.arange(200),'tokenized_prompt_mask':mask,
                'token_ar_mask':np.arange(200),'state':np.arange(32)}

    def test_preserves_internal_masked_tokens_and_valid_positions(self):
        inputs=self.inputs();result=codec.compact_inputs(inputs)
        np.testing.assert_array_equal(result['tokenized_prompt'],np.arange(64))
        np.testing.assert_array_equal(result['tokenized_prompt_mask'][:3],[True,False,True])
        np.testing.assert_array_equal(result['token_ar_mask'],np.arange(64))
        self.assertIs(result['state'],inputs['state'])
        self.assertEqual(len(inputs['tokenized_prompt']),200)

    def test_never_drops_active_or_unknown_third_view(self):
        for mask in ({},{'right_wrist_0_rgb':True}):
            inputs=self.inputs();inputs['image_mask']=mask
            with self.assertRaisesRegex(ValueError,'masked right-wrist'):codec.compact_inputs(inputs)

    def test_rejects_inconsistent_token_arrays(self):
        inputs=self.inputs();inputs['tokenized_prompt']=np.arange(3)
        with self.assertRaisesRegex(ValueError,'length mismatch'):codec.compact_inputs(inputs)

    def test_empty_prompt_keeps_a_stable_masked_bucket(self):
        inputs=self.inputs();inputs['tokenized_prompt_mask'][:]=False
        result=codec.compact_inputs(inputs)
        self.assertEqual(len(result['tokenized_prompt']),64)
        self.assertFalse(result['tokenized_prompt_mask'][0])

    def test_long_prompts_never_lose_valid_tokens(self):
        for last,size in ((63,64),(64,128),(127,128),(128,200),(199,200)):
            inputs=self.inputs();inputs['tokenized_prompt_mask'][last]=True
            result=codec.compact_inputs(inputs)
            self.assertEqual(len(result['tokenized_prompt']),size)
            self.assertTrue(result['tokenized_prompt_mask'][last])

if __name__=='__main__':unittest.main()
