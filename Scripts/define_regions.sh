
#!/usr/bin/bash

Set_areas(){

	awk '{

	strand=$5;
	Chrom=$1;
	start=$2;
	end=$3;
	name=$4;
	
	if (strand=="+"){

		new_start = start - 400;

			if (new_start <0){
				new_start =0;
			}

		new_end = start + 125;
		print Chrom "\t" new_start "\t" new_end "\t" name "\t" 1 "\t" strand;
		}

	else if (strand=="-"){
		new_start = end + 400;
		new_end = end - 125;
		print Chrom "\t" new_end "\t" new_start "\t" name "\t" 1 "\t" strand;
		}
	}' $1
}

Set_areas $1 > output.bed

bedtools getfasta -s -fi ./sacCer3_genome.fa -bed output.bed -name > $2

rm output.bed
echo "Generated Sequences!"
