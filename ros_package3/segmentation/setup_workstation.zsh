#!/usr/bin/env zsh

for file in */
do
	echo "$file" >> filenames.txt
done

grep sub/ filenames.txt > /dev/null
is_in=$?

if [ "$is_in" != 0 ]
then
	echo "Creating sub"
	mkdir sub
fi

grep segmentation/ filenames.txt > /dev/null
is_in=$?

if [ "$is_in" != 0 ]
then
	echo "FileNotFound <segmentation>. Please check if directory exists."
	exit 1
fi

# shellcheck disable=SC2164
cd segmentation

for file in *.py
do
	echo "$file" >> filenames.txt
done

to_exit=0
for filename in all_together.py global_registration2.py PointOfViewCamera.py RANSAC.py
do
	grep $filename filenames.txt > /dev/null
	is_in=$?

	if [ "$is_in" != 0 ]
	then
		echo "FileNotFound. Cannot find ${filename} in segmentation. Please check if file exists."
		to_exit=1
	fi
done

rm filenames.txt

# shellcheck disable=SC2103
cd ..

if [ $to_exit -eq 1 ]
then
	exit 1
fi

rm filenames.txt

echo 'Setup complete. No issues found'
