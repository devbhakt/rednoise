#!/bin/bash


bd=/Users/mkeith/Code/run_enterprise/

par=$1
tim=$2

psr=$(grep PSRJ $par | awk '{print $2}')

dm=$(grep be $tim | awk '{if ($2 < 1200 || $2 > 2000) print "DM"}' | head -n 1)

$bd/disableparam.py $par START FINISH > $par.tmp
$bd/enableparam.py $par.tmp JUMP RAJ DECJ F0 F1 PX PB A1 OM ECC T0 PMRA PMDEC $dm DM1 DM2 -glitch > ${psr}_first.par
echo "F2 0 1" >> ${psr}_first.par
if grep -q "PMDEC" $par ; then
    echo "PM already included"
else
    echo "PMRA 0 1" >> ${psr}_first.par
    echo "PMDEC 0 1" >> ${psr}_first.par
fi

redmax=-7

#$bd/run_enterprise.py ${psr}_first.par $tim -N 1e5 --Ared-max $redmax --red-prior-log --plot-chain
mv ${psr}.corner.pdf ${psr}.corner_first.pdf
mv ${psr}.chain.pdf ${psr}.chain_first.pdf

tempo2 -f ${psr}_first.par.post $tim -outpar ${par}.tmp2 -qrfit

f2=$(grep -e '^F2 ' ${par}.tmp2 | awk '{print $2}')
f2e=$(grep -e '^F2 ' ${par}.tmp2 | awk '{print $4}')
f2range=$(echo $f2 $f2e | awk '{print $1+6*$2}')

pmrange=$(grep -e '^PMDEC ' ${par}.tmp2 | awk '{print $4*20}')

redmax=$(grep TNRedAmp ${psr}_first.par.results | awk '{print 1+$9}')


$bd/disableparam.py $par.tmp2 PMRA PMDEC | grep -v F2 > ${psr}_run.par

rm $par.tmp $par.tmp2

f0=$(grep -e '^F0 ' $par | awk '{print $2}')
f1=$(grep -e '^F1 ' $par | awk '{print $2}')

echo "F2 range is: $f2range"
echo "PM range is: $pmrange"
echo "Red Max is : $redmax"


#tempo2 -f $par.m $tim -qrfit -outpar ${psr}_run.par

$bd/run_enterprise.py ${psr}_run.par $tim -j --plot-chain --all-corner --red-ncoeff 60 -N 1e5 -M "--Ared-max $redmax --pm --pm-range $pmrange --f2 $f2range" \
    "--Ared-max $redmax --pm --pm-range $pmrange" \
    "--no-red-noise --pm --pm-range $pmrange --f2 $f2range" \
    "--no-red-noise --pm --pm-range $pmrange"

