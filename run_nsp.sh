#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=0

DOCDIR=zeshel/documents

train_domains=("american_football" "doctor_who" "fallout" "final_fantasy" "military" "pro_wrestling" "starwars" "world_of_warcraft")
val_domains=("coronation_street" "elder_scrolls" "ice_hockey" "muppets")
test_domains=("forgotten_realms" "lego" "star_trek" "yugioh")

concat() {
  documents=""
  for domain in $@;
  do documents=${documents:+$documents}$DOCDIR/${domain}.json,;
  done
  documents=${documents::-1}
	echo $documents
}

train_documents="$(concat ${train_domains[@]})"
val_documents="$(concat ${val_domains[@]})"
test_documents="$(concat ${test_domains[@]})"
