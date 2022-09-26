#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/yinxr/hx/liangshihao/pretrained_models/bert-base/"

if [ -z $DOCDIR]; then
  DOCDIR="data/documents"
fi

if [ -z $MENTIONDIR]; then
  MENTIONDIR="data/random_padded_mentions/"
fi

if [ -z $TFIDFDIR]; then
  TFIDFDIR="data/random_padded_tfidfs/"
fi

domains=("american_football" "doctor_who" "fallout" "final_fantasy" "military" "pro_wrestling" "starwars" "world_of_warcraft" \
"coronation_street" "elder_scrolls" "ice_hockey" "muppets" "forgotten_realms" "lego" "star_trek" "yugioh")

concat() {
  documents=""
  for domain in $@;
  do documents=${documents:+$documents}$DOCDIR/${domain}.json,;
  done
  documents=${documents::-1}
	echo $documents
}

documents="$(concat ${domains[@]})"

python3 main.py --document-file $documents \
                --train-mentions-file $MENTIONDIR/train.json \
                --eval-mentions-file $MENTIONDIR/valid.json \
                --train-tfidf-candidates-file $TFIDFDIR/train_tfidfs.json \
                --eval-tfidf-candidates-file  $TFIDFDIR/valid_tfidfs.json \
                --pretrained-model-path $MODEL_PATH

