from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Chunk:
    chunk_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Drug:
    drug_id: str
    name_ko: str
    name_en: Optional[str]
    ingredient: List[str]
    indications: str
    dosage: str
    warnings: str
    category: Optional[str]
    updated_date: Optional[str]
    combination_contraindication: List[str]
    parent_text: str
    child_chunks: List[Chunk] = field(default_factory=list)


@dataclass
class Symptom:
    symptom_id: str
    name: str
    is_red_flag: Optional[bool]
    urgency: Optional[str]
    context: Optional[str]
    action_guide: Optional[str]


@dataclass
class UserInput:
    symptoms: List[str]
    current_drug_ids: List[str] = field(default_factory=list)
    current_drug_names: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    age: Optional[int] = None


@dataclass
class SafetyResult:
    has_red_flag: bool
    red_flag_symptoms: List[str] = field(default_factory=list)
    interaction_warnings: List[str] = field(default_factory=list)
    contraindication_warnings: List[str] = field(default_factory=list)
    general_warnings: List[str] = field(default_factory=list)


@dataclass
class ChromaEvidence:
    chunk_id: str
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0
    relevance: float = 0.0


@dataclass
class RetrievalResult:
    drug: Drug
    score: float
    matched_symptoms: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    chroma_evidence: List[ChromaEvidence] = field(default_factory=list)
    de_score: Optional[float] = None
    retrieval_source: str = "legacy_json"
    de_document_snippet: Optional[str] = None
    catalog_match: str = "catalog"


class ConversationPhase(str, Enum):
    GREETING = "greeting"
    COLLECTING = "collecting"
    CLARIFYING = "clarifying"
    RECOMMENDING = "recommending"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    ENDED = "ended"


@dataclass
class ConversationTurn:
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class ConversationSession:
    """multi-turn 대화 상태."""

    user_input: UserInput = field(default_factory=lambda: UserInput(symptoms=[]))
    phase: ConversationPhase = ConversationPhase.GREETING
    turns: List[ConversationTurn] = field(default_factory=list)
    confirmed_slots: set[str] = field(default_factory=set)
    pending_slot: Optional[str] = None
    turn_count: int = 0
    last_pipeline_summary: Optional[str] = None
    has_recommendation: bool = False # 💡 추가: 추천이 발생했는지 여부 저장


@dataclass
class PipelineResult:
    """enhanced pipeline 구조화 결과."""

    user_input: UserInput
    trace_steps: List[Any]
    decision: Any
    validation: Any
    is_emergency: bool = False
    emergency_message: str = ""
    symptom_context: List[ChromaEvidence] = field(default_factory=list)
    data_provenance: Optional[Any] = None
    reasoning_trace: Optional[Dict[str, Any]] = None


@dataclass
class ConversationResponse:
    """한 턴의 어시스턴트 응답."""

    message: str
    session: ConversationSession
    phase: ConversationPhase
    debug_trace: Optional[str] = None
    data_sources: Optional[Dict[str, Any]] = None
    reasoning_trace: Optional[Dict[str, Any]] = None