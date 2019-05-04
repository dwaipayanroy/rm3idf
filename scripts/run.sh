#!/bin/bash
# RM3IDF proposed in:
# "Selecting Discriminative Terms for Relevance Model" --- SIGIR 2019
# Dwaipayan Roy, Sumit Bhatia and Mandar Mitra.

cd ../

homepath=`eval echo ~$USER`
stopFilePath="$homepath/smart-stopwords"
if [ ! -f $stopFilePath ]
then
    echo "Please ensure that the path of the stopword-list-file is set in the .sh file."
else
    echo "Using stopFilePath="$stopFilePath
fi

if [ $# -le 7 ] 
then
    echo "Usage: " $0 " <following arguments in the order>";
    echo "1. Path of the index.";
    echo "2. Path of the query.xml file."
    echo "3. Path of the directory to store res file."
    echo "4. Number of expansion documents";
    echo "5. Number of expansion terms";
    echo "6. RM3 - QueryMix:";
    echo "7. RM3-IDF - 1/2/3 (any other value will set vanilla RM3)";
    echo "8. Similarity Function: 2: LM-JM, 3: LM-Dir";
    exit 1;
fi

indexPath=`readlink -f $1`		# absolute address of the index
queryPath=`readlink -f $2`		# absolute address of the query file
resPath=`readlink -f $3`		# absolute directory path of the .res file
resPath=$resPath"/"

queryName=$(basename $queryPath)
prop_name="rm3idf-"$7"-"$queryName".D-"$4".T-"$5".properties"

echo "Using index at: "$indexPath
echo "Using query at: "$queryPath
echo "Using directory to store .res file: "$resPath

fieldToSearch="content"
fieldForFeedback="content"

rm3_idf=$7

echo "Field for searching: "$fieldToSearch
echo "Field for feedback: "$fieldForFeedback

similarityFunction=$8

case $similarityFunction in
    2) param1=0.2
       param2=0.0 ;;
    3) param1=500
       param2=0.0 ;;
esac

echo "similarity-function: "$similarityFunction" " $param1

# making the .properties file
cat > $prop_name << EOL

indexPath=$indexPath

fieldToSearch=$fieldToSearch

fieldForFeedback=$fieldForFeedback

queryPath=$queryPath

stopFilePath=$stopFilePath

resPath=$resPath

numHits= 1000

similarityFunction=$similarityFunction

param1=$param1
param2=$param2

# Number of documents
numFeedbackDocs=$4

# Number of terms
numFeedbackTerms=$5

rm3.queryMix=$6

rm3.idf=$rm3_idf

qrelPath=$qrelPath

toTRF=$toTRF

feedbackFromFile=false

feedbackFilePath=$feedbackFilePath

EOL
# .properties file made

java -Xmx1g -cp $CLASSPATH:dist/RM3IDF.jar RelevanceFeedback.RelevanceBasedLanguageModel $prop_name

