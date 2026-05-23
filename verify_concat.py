import csv
import os
import soundfile as sf

# Verify a few samples from each split
for split in ['train', 'dev', 'test']:
    tsv_path = rf'd:\Codes\train_whisper_cpu\cv_zhTW_concat\{split}\data.tsv'
    clips_dir = rf'd:\Codes\train_whisper_cpu\cv_zhTW_concat\{split}\clips'
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    
    print(f'=== {split} ===')
    print(f'  Total samples: {len(rows)}')
    
    # Check first 3 samples
    for i in [0, len(rows)//2, -1]:
        row = rows[i]
        audio_path = os.path.join(clips_dir, row['path'])
        audio, sr = sf.read(audio_path, dtype='float32')
        duration = len(audio) / sr
        sentence_preview = row['sentence'][:80].replace('\n', ' | ')
        print(f'  [{row["path"]}] dur={duration:.2f}s, clips={row["num_clips"]}, text="{sentence_preview}..."')
    
    # Check total size
    total_size = sum(os.path.getsize(os.path.join(clips_dir, r['path'])) for r in rows) / (1024*1024)
    print(f'  Total audio size: {total_size:.1f} MB')
    print()