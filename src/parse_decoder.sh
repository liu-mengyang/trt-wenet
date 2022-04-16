#!/usr/bin/env bash
python surgeon_decoder.py

trtexec \
	--onnx=decoder_sed.onnx \
	--explicitBatch \
	--minShapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
	--optShapes=encoder_out:16x64x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
	--maxShapes=encoder_out:64x256x256,encoder_out_lens:64,hyps_pad_sos_eos:64x10x64,hyps_lens_sos:64x10,ctc_score:64x10 \
	--saveEngine=decoder.plan \
	--workspace=40960 \
	--buildOnly \
	--verbose \

