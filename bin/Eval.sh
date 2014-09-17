#!/bin/bash

echo 'Evaluating...'

while getopts ":g:e:u" opt; do
  case $opt in
    g)
      GOLD_FILE=$OPTARG
      echo "Using Gold File: $OPTARG" >&2
      ;;
    e)
      PARSED_FILE=$OPTARG
      echo "Using Parsed File: $OPTARG" >&2
      ;;
    u)
      UNLABELED=true
      echo "Using Parsed File: $OPTARG" >&2
      ;;
    \?)
      echo "ERROR: Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "ERROR: Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ -z "$GOLD_FILE" ]]; then
  echo "ERROR: Gold File Required! (Option -g)"
  exit 1
fi

if [[ -z "$PARSED_FILE" ]]; then
  echo "ERROR: Parsed File Required! (Option -e)"
  exit 1
fi

if [[ $UNLABELED ]]; then
  echo "Evaluating with Unlabeled Trees"
  python bin/parseval.py -u $PARSED_FILE $GOLD_FILE
else
  echo "Evaluating with Labeled Trees"
  python bin/parseval.py $PARSED_FILE $GOLD_FILE 
fi
