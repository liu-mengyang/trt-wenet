#!/usr/bin/env bash
python surgeon_encoder.py

trtexec \
	--onnx=encoder_sed.onnx \
	--explicitBatch \
	--minShapes=speech:1x16x80,speech_lengths:1 \
	--optShapes=speech:16x64x80,speech_lengths:16 \
	--maxShapes=speech:64x256x80,speech_lengths:64 \
	--saveEngine=encoder.plan \
	--workspace=20000 \
	--buildOnly \
	--fp16 \
	--verbose \

