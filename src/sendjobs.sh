#!/bin/bash
# 
# Send all my jobs in a swarm

L=15
n=15
#Q=5
for n in 2 3 4 5 6 7 8 9 10 15 20
do
	jobfile="runjob.sh"
	# let L=LL*PP
	# let P=PP*10
	let L1=L-1
	let L2=L+2
	let L3=P-1

	echo "submitting with N=$n"

	echo "#!/bin/sh" > $jobfile
	echo "#SBATCH --account=theory" >> $jobfile
	echo "#SBATCH --job-name=geom" >> $jobfile
	echo "#SBATCH -c 1" >> $jobfile
	echo "#SBATCH --time=1-11:59:00" >> $jobfile
	echo "#SBATCH --mem-per-cpu=1gb" >> $jobfile

	echo "module load anaconda/3-5.1" >> $jobfile
	echo "source activate pete" >> $jobfile
	
	echo "python repler/habaexperiment.py -n $n" >> $jobfile

	echo "date" >> $jobfile

	sbatch $jobfile
	echo "waiting"
	sleep 1s
done

