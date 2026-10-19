"""Independent append-only exact lookup prototype; NOT used by the engine.

All token IDs are known before append. One exact3gram key maps to ordered
uint32 occurrence positions; query verifies actual longer suffixes. Dense
repetition can still require a long candidate scan. No rollback/writer/CUDA.
"""
from array import array
import time


UINT32_MASK=(1<<32)-1
MODES=('ambiguity','source-boundary','both')


class ExactNgramIndex:
    """Bounded known-history index; no source text, target logits or outcomes.

`append` owns its input IDs. A source ending at position p is registered only
when the next token at p becomes known. Thus the current suffix never matches
itself and all source continuations exist within already-known history.
"""
    def __init__(self,history=(),*,max_history_tokens=131072):
        if type(max_history_tokens) is not int or not 0<=max_history_tokens<=131072:
            raise ValueError('Prototype capacity must be0..131072 known tokens')
        self._maximum=max_history_tokens;self._history=[]
        self._positions={};self._suffix3=0
        self.append(history)

    @property
    def history_length(self):return len(self._history)

    def append(self,tokens):
        """Validate all IDs/capacity before mutation, then add true known IDs."""
        values=tuple(tokens)
        if len(self._history)+len(values)>self._maximum:
            raise ValueError('Known-history prototype capacity exceeded')
        if any(type(token) is not int or not 0<=token<=UINT32_MASK for token in values):
            raise ValueError('Require uint32 Python token IDs')
        for token in values:
            p=len(self._history)
            if p>=3:
                positions=self._positions.get(self._suffix3)
                if positions is None:
                    positions=array('I');self._positions[self._suffix3]=positions
                positions.append(p)
            self._suffix3=((self._suffix3<<32)|token)&((1<<96)-1)
            self._history.append(token)

    def _match(self,with_statistics=True):
        if len(self._history)<4:return None
        positions=self._positions.get(self._suffix3)
        if positions is None:return None
        for n in range(min(16,len(self._history)-1),2,-1):
            suffix=self._history[-n:];count=0;latest=None;next_ids=set()
            for end in reversed(positions):
                if end>=n and self._history[end-n:end]==suffix:
                    if not with_statistics:return n,end,1,1
                    if latest is None:latest=end
                    count+=1;next_ids.add(self._history[end])
            if count:return n,latest,count,len(next_ids)
        return None

    def draft(self,max_tokens=127):
        """Exact legacy proposal IDs, no new policy metadata or fallback."""
        if type(max_tokens) is not int or not 0<=max_tokens<=127:
            raise ValueError('Proposal capacity must be0..127')
        match=self._match(False) if max_tokens else None
        if match is None:return []
        end=match[1];return list(self._history[end:end+max_tokens])

    def select(self,mode,max_tokens,*,prompt_tokens):
        start=time.perf_counter();length=len(self._history)
        if mode not in MODES or type(max_tokens) is not int or not 0<=max_tokens<=127:
            raise ValueError('Require active known mode and capacity0..127')
        if type(prompt_tokens) is not int or not 0<=prompt_tokens<=length:
            raise ValueError('Invalid known prompt boundary')
        metadata=dict(mode=mode,known_history_tokens=length,prompt_tokens=prompt_tokens,
            matched_ngram=0,matching_sources=0,distinct_next_ids=0,selected_source_end=None,
            source_segment_remaining=None,original_draft_tokens=0,selected_draft_tokens=0,
            crosses_prompt_output_boundary=False,ambiguity_scout=False,source_boundary_capped=False,
            action='zero_budget' if max_tokens==0 else 'no_match',
            decision_scope='Computed before target forward; acceptance is never a selection input')
        match=self._match() if max_tokens else None;selected=[]
        if match is not None:
            n,end,count,distinct=match;original=list(self._history[end:end+max_tokens])
            remaining=prompt_tokens-end if end<prompt_tokens else length-end
            metadata.update(matched_ngram=n,matching_sources=count,distinct_next_ids=distinct,
                selected_source_end=end,source_segment_remaining=remaining,original_draft_tokens=len(original),
                crosses_prompt_output_boundary=end<prompt_tokens<end+len(original))
            if mode in ('ambiguity','both') and distinct>1:
                metadata.update(ambiguity_scout=True,action='M1_ambiguity_scout')
            else:
                selected=original
                if mode in ('source-boundary','both') and remaining<len(original):
                    selected=original[:remaining];metadata.update(source_boundary_capped=True,action='source_boundary_cap')
                else:metadata['action']='original_draft'
        metadata['selected_draft_tokens']=len(selected);rows=len(selected)+1
        metadata['chosen_verifier_rows']=rows
        metadata['shape_matches_HC_32_64_128_preset_or_M1']=rows in (1,32,64,128)
        metadata['shape_hint_scope']='Requested shape compatibility only, not proof of actual graph hits'
        metadata['real_rows_per_M32_tile_if_enabled']=[32]*(rows//32)+([rows%32] if rows%32 else [])
        metadata['selection_cpu_s']=time.perf_counter()-start
        return selected,metadata

    def status(self):
        return dict(protocol='append_only_exact_3gram_occurrences_prototype_v1',history_tokens=len(self._history),
            maximum_history_tokens=self._maximum,min_ngram=3,max_ngram=16,
            unique_3gram_entries=len(self._positions),occurrence_positions=sum(len(value) for value in self._positions.values()),
            equality='Exact96bit3gram keys and actual3..16 token window comparison; no fingerprint collisions',
            concurrent_writer_supported=False,rollback_supported=False,production_integrated=False,
            query_complexity='Worst14 scans of matching3gram occurrence positions; dense repetitions can degrade',
            memory_scope='Entry counts only; OS peak RSS/working set must be measured separately')
