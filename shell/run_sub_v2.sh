#!/bin/bash
cd /root
rm -rf /root/submission_final
/root/miniconda3/bin/python make_testA_submission.py --out_dir /root/submission_final > /root/logs/make_submission.log 2>&1
