#!/usr/bin/zsh
while read -u 3 p 
do
		echo "$p"
		echo "add above line y/n"
		read -q answer  
		if [[ $answer =~ ^[Yy]$ ]]
		then
				echo "$p" >> subset.json
		fi
done 3< variables.json
