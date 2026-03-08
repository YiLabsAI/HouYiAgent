from houyi.adapters.memory import MemoryRecord, MemoryScope, MemoryStore
from houyi.adapters.memory.store import MemoryStore as ExportedMemoryStore
from houyi.adapters.memory.types import MemoryRecord as ExportedMemoryRecord
from houyi.adapters.memory.types import MemoryScope as ExportedMemoryScope


def test_memory_adapter_exports_canonical_symbols() -> None:
    assert MemoryStore is ExportedMemoryStore
    assert MemoryRecord is ExportedMemoryRecord
    assert MemoryScope is ExportedMemoryScope
