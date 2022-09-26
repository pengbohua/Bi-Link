#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7


if [ -z $DOCDIR]; then
  DOCDIR="data/documents"
fi

if [ -z $MENTIONDIR]; then
  MENTIONDIR="data/mentions"
fi

if [ -z $TFIDFDIR]; then
  TFIDFDIR="data/tfidf_candidates"
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
                --train-mentions-file data/trunc_mentions/train.json \
                --eval-mentions-file $MENTIONDIR/valid.json \
                --train-tfidf-candidates-file data/trunc_tfidfs/train_tfidfs.json \
                --eval-tfidf-candidates-file  $TFIDFDIR/valid_tfidfs.json \
                --train-batch-size 64 \
                --eval-batch-size 64 \

