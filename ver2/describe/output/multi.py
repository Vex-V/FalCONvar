"""One description, several destinations.

Identical policy to ingest's ``MultiSink``, and literally the same code for
the part that matters -- first sink authoritative, a failing secondary
reported once and dropped. See ``ver2.fanout.FanOut``, which both stages
share so that neither has to import the other.

``existing`` returns what **every** sink already holds, not what the primary
holds. Asking only the primary was tried and is wrong: a secondary that was
dropped mid-run, or that lost rows some other way, can then never catch up --
the primary reports the work as done and nothing ever repairs the other copy.
Writing to two destinations is only worth anything if both end up complete.

The cost is that a gap in any copy is refilled by describing the pair again,
which is the expensive call. That is the right trade because gaps should be
rare, and the alternative is a second copy that is quietly and permanently
missing rows while reporting success.
"""

from __future__ import annotations

from typing import Any, Optional

from ver2.fanout import FanOut


class MultiDescriptionSink(FanOut):
    """Fans descriptions out to several sinks. A ``DescriptionSink`` itself."""

    def begin(self, video_id: str, manifest: dict[str, Any], model: dict[str, Any]) -> None:
        self.primary.begin(video_id, manifest, model)
        for sink in list(self.secondary):
            self._try(sink, lambda s: s.begin(video_id, manifest, model), "begin")

    def existing(self) -> set[tuple[int, str]]:
        done = self.primary.existing()
        for sink in list(self.secondary):
            # A secondary that cannot answer is dropped, not trusted: treating
            # its silence as "has everything" is what would hide the gap.
            before = set(done)
            self._try(sink, lambda s: done.intersection_update(s.existing()),
                      "existing")
            if sink not in self.secondary:
                done = before
        return done

    def described(self, record: dict[str, Any]) -> None:
        self.primary.described(record)
        for sink in list(self.secondary):
            self._try(sink, lambda s: s.described(record),
                      f"chunk {record.get('chunk_id')}/{record.get('sampler')}")

    def finish(self, stats: Optional[dict] = None) -> dict[str, Any]:
        for sink in list(self.secondary):
            self._try(sink, lambda s: s.finish(stats), "finish")
        return self.primary.finish(stats)
