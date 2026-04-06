from __future__ import annotations


class ResearchRuntimeError(RuntimeError):
    pass


class ResearchPlanMissingError(ResearchRuntimeError):
    pass


class ResearchStateError(ResearchRuntimeError):
    pass


class ResearchTimeoutError(ResearchRuntimeError):
    pass


class ResearchCancelledError(ResearchRuntimeError):
    pass


class ResearchReportNotReadyError(ResearchRuntimeError):
    pass
