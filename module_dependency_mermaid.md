
```mermaid
graph TB
    subgraph Configuration
        C1[PIPER_BIN / PIPER_MODEL / PIPER_MAX_CHARS]
        C2[logging.basicConfig]
    end

    subgraph AppState["AppState (State Management)"]
        S1[lines_cache / cache_lock]
        S2[stop_event threading.Event]
        S3[current_thread / current_procs]
        S4[now_playing + nonce]
        S5[saved_idx / saved_sentences]
    end

    subgraph FileOps["File Operations"]
        F1[_cache_key_for_file]
        F2[_get_lines_cached]
    end

    subgraph ChunkMgmt["Chunk Management"]
        K1[get_chunks]
        K2[show_chunk]
    end

    subgraph TextProc["Text Processing"]
        T1[_preprocess_text]
        T1a["skip ** lines"]
        T1b["remove **bold** tags"]
        T1c["remove ** to end-of-line"]
        T2[_split_into_sentences]
        T2a["no split inside quotes"]
        T2b["split by sentence boundary"]
    end

    subgraph TTS["TTS Engine"]
        E1[_split_text_into_chunks]
        E2[_run_piper_to_paplay]
        E2a[subprocess.Popen piper]
        E2b[subprocess.Popen paplay]
        E3[_tts_worker]
        E3a["play from start_idx"]
        E3b["update now_playing"]
        E3c[save_progress]
        E3d[clear_progress]
    end

    subgraph UI["UI Handlers"]
        U1[stop_tts]
        U1a[request_stop]
        U1b[terminate_processes]
        U2[read_english_start]
        U2a[get_resume_info]
        U2b[launch _tts_worker]
        U3[get_now_playing]
    end

    subgraph Gradio["Gradio 5 UI"]
        G1[gr.File Upload File]
        G2[gr.Number Chunk Size]
        G3[gr.Radio Select Chunk]
        G4[gr.Textbox File Content]
        G5[gr.Textbox Now Playing]
        G6[gr.Button Read English / Stop]
        G7[gr.Timer 100ms]
    end

    subgraph External["External Tools"]
        X1[Piper TTS synthesis]
        X2[Paplay PulseAudio play]
    end

    Configuration -.->|global config| FileOps
    Configuration -.->|global config| AppState
    Configuration -.->|global config| ChunkMgmt
    Configuration -.->|global config| TTS
    Configuration -.->|global config| TextProc

    FileOps <--> AppState
    ChunkMgmt -->|chunk text| TextProc
    TextProc -->|sentences[]| TTS
    AppState -.->|read/write state| TTS
    TTS -->|stdin/stdout| External
    UI -->|control| AppState
    UI -->|start thread| TTS
    Gradio -->|click events| UI
    Gradio -.->|show_chunk| ChunkMgmt
    Gradio -.->|upload| FileOps
    Gradio -.->|Timer polling| get_now_playing

    T1 --> T1a
    T1 --> T1b
    T1 --> T1c
    T2 --> T2a
    T2 --> T2b
    E2 --> E2a
    E2 --> E2b
    E3 --> E3a
    E3 --> E3b
    E3 --> E3c
    E3 --> E3d
    U1 --> U1a
    U1 --> U1b
    U2 --> U2a
    U2 --> U2b
    G7 -->|timer.tick| U3
```
