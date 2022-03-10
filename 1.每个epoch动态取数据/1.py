# -*-coding:utf-8-*-
from Bio import SeqIO

with open("data.txt","w") as write:
    for seq_record in SeqIO.parse("soil-32s-ARGs.faa","fasta"):
        write.write(str(seq_record.seq) +"\t"+ seq_record.id +"\n")