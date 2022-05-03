#!/usr/bin/env bash
python surgeon_encoder.py

trtexec \
	--onnx=encoder_sed.onnx \
	--explicitBatch \
	--minShapes=speech:1x16x80,speech_lengths:1 \
	--optShapes=speech:16x64x80,speech_lengths:16 \
	--maxShapes=speech:64x256x80,speech_lengths:64 \
	--saveEngine=encoder.plan \
	--plugins=/target/LayerNormPlugin.so \
	--workspace=40960 \
	--buildOnly \
	--noTF32 \
	--verbose \

