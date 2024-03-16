# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:06:34 2024

@author: 1
"""
import sys
import time
import mido
from mimix.predictor import LMGenerator
from mimix.utils import real_path, load_model_config


def convert_tokens_to_midi(tokens, output_path):
    """
    """
    new_midi_file = mido.MidiFile(ticks_per_beat=384)
    new_track = mido.MidiTrack()
    new_midi_file.tracks.append(new_track)
    new_midi_file.tracks[0].append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    new_midi_file.tracks[0].append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    
    def add_event(track, kv):
        """
        """
        kv = {k:int(kv[k]) if k != "type" else kv[k] for k in kv}
        if len(kv) == 0:
            pass
        elif kv["type"] == "note_on":
            track.append(mido.Message('note_on', note=kv["note"], velocity=kv["velocity"], time=kv["time"]))
        elif kv["type"] == "control_change":
            track.append(mido.Message('control_change', control=kv["control"], value=kv["value"], time=kv["time"]))
        elif kv["type"] == "program_change":
            track.append(mido.Message('program_change', program=int(kv["program"])))
    
    event = {}
    for token in tokens:
        
        if token.startswith("note_on_note_"):
            add_event(new_midi_file.tracks[0], event)
            event = {}
            event["type"] = "note_on"
            k = "note"
            event[k] = token.split("_")[-1]
        elif token.startswith("note_on_velocity_"):
            k = "velocity"
            event[k] = token.split("_")[-1]
        elif token.startswith("note_on_time_"):
            k = "time"
            event[k] = token.split("_")[-1]
        elif token.startswith("control_change_control_"):
            add_event(new_midi_file.tracks[0], event)
            event = {}
            event["type"] = "control_change"
            k = "control"
            event[k] = token.split("_")[-1]
        elif token.startswith("control_change_time_"):
            k = "time"
            event[k] = token.split("_")[-1]
        elif token.startswith("control_change_value_"):
            k = "value"
            event[k] = token.split("_")[-1]
        elif token.startswith("program_change_program_"):
            add_event(new_midi_file.tracks[0], event)
            event["type"] = "program_change"
            event = {}
            k = "program"
            event[k] = token.split("_")[-1]
        elif token.startswith("#"):
            event[k] = event[k] + token[1:]
    new_midi_file.tracks[0].append(mido.MetaMessage('end_of_track', time=0))
    new_midi_file.save(output_path)


def midi_demo():
    """
    """
    conf_file = "conf/midi_base_conf"
    config = load_model_config(real_path(conf_file))
    lm_gen = LMGenerator(config)
     
    print("Press to start midi generation.")
    for line in sys.stdin:

        start = time.time()
        search_res = lm_gen.predict(prefix_list=None) 
        tokens = search_res[0][1][0][0].split()
        convert_tokens_to_midi(tokens, "test.mid")
        
        print("Generate midi and save to test.mid done.")
        end = time.time()
        cost = end - start
        print("-----cost time: %s s-----" % cost)
        print("Press to start midi generation.")
        

if __name__ == "__main__":
    
    midi_demo()