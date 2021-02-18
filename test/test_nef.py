#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:49:15 2021

@author: mdeprez
"""

import argparse
from bayesNN import SNP_bnn
import time
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

### % Argument parser

parser = argparse.ArgumentParser(description='Compute accuracy for some bnns')


parser.add_argument(
	'-ns',
	'--nbsubject',
	type=int,
	default=500,
	help='number of subjects'
)

parser.add_argument(
	'-g',
	'--genes',
	type=int,
	default=4,
	help='number of genes'
)

parser.add_argument(
	'-t',
	'--target',
	type=int,
	default=11,
	help='number of target dimensions'
)

parser.add_argument(
	'-r',
	'--relevant',
	type=int,
	default=1,
	help='number of relevant genes'
)

parser.add_argument(
	'-n',
	'--noise',
	type=float,
	default=0.05,
	help='amount of noise'
)

args = parser.parse_args()

## Run BNN
start_time = time.time()
data=SNP_bnn.gfptoCSV(args.nbsubject, args.genes, args.target, args.relevant, noise=args.noise)
bnn=SNP_bnn(data["X"], data["Y"])
bnn.optimize(data["X"], data["Y"], epochmax=100000, step=1000)
end_time = time.time()

# Ground truth relevant genes
y = [0]*args.relevant + [1]*(args.genes - args.relevant)

# Get perfomance metrics
compil_time = end_time - start_time
precision, recall, thresholds = precision_recall_curve(y,  bnn.probalpha().tolist())
auc = auc(recall, precision)

values_name = ["compilation_time", "precision", "recall", "thresholds", "Auc"]
print("test")
print("\t".join(values_name))
for i in range(0, len(precision)-1):
    print("\t".join(map(str, [round(compil_time,2), round(precision[i],2), round(recall[i],2), thresholds[i], auc])))
