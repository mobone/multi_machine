#!/usr/bin/env bash
cd /home/nick/multi_machine
exec bash
for run in {1..24}
do
    rq worker --url redis://192.168.1.127 &
done