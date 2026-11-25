#!/usr/bin/env python3
"""Train chronological multicity quantile model and publish validated artifacts."""
import argparse, json
from aircast.probabilistic import train

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--artifacts', default=None)
    args=parser.parse_args()
    print(json.dumps(train(args.data,args.artifacts),indent=2))
