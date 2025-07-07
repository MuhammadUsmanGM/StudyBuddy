from pydantic import BaseModel
from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, Runner, output_guardrail
from my_secrets import Secrets

secrets = Secrets()

class MessageOutput(BaseModel):
    response: str


# 1. PII
class PIICheckOutput(BaseModel):
    contains_pii: bool
    is_developer_context: bool
    reasoning: str

pii_agent = Agent(
    name="PII Guardrail",
    instructions="""
Detect if the output contains PII like names, emails, or phone numbers.
Ignore mock data, placeholders, and internal developer examples.
Return:
- contains_pii: True/False
- is_developer_context: True if it's developer-focused content.
""",
    output_type=PIICheckOutput,
    model=secrets.gemini_api_model,
)

@output_guardrail
async def pii_output_guardrail(ctx: RunContextWrapper, agent: Agent, output: MessageOutput) -> GuardrailFunctionOutput:
    result = await Runner.run(pii_agent, output, context=ctx.context)
    data = result.final_output
    return GuardrailFunctionOutput(output_info=data, tripwire_triggered=data.contains_pii and not data.is_developer_context)

# 2. Hallucination
class HallucinationCheckOutput(BaseModel):
    is_factually_inaccurate: bool
    is_developer_context: bool
    reasoning: str

hallucination_agent = Agent(
    name="Factual Accuracy Guardrail",
    instructions="""
Check if the response contains fabricated or unverified facts.
Do not flag fictional examples or developer test strings.
Return:
- is_factually_inaccurate: True/False
- is_developer_context: True if it's a dev/test context.
""",
    output_type=HallucinationCheckOutput,
    model=secrets.gemini_api_model,
)

@output_guardrail
async def hallucination_output_guardrail(ctx: RunContextWrapper, agent: Agent, output: MessageOutput) -> GuardrailFunctionOutput:
    result = await Runner.run(hallucination_agent, output, context=ctx.context)
    data = result.final_output
    return GuardrailFunctionOutput(output_info=data, tripwire_triggered=data.is_factually_inaccurate and not data.is_developer_context)


# 3. Self-Reference
class SelfReferenceCheckOutput(BaseModel):
    contains_self_reference: bool
    is_developer_context: bool
    reasoning: str

self_reference_agent = Agent(
    name="Self-Reference Guardrail",
    instructions="""
Detect statements referring to the AI model itself (e.g., "As an AI model...").
Ignore developer debug logs or internal technical references.
Return:
- contains_self_reference: True/False
- is_developer_context: True if it's a dev-related explanation.
""",
    output_type=SelfReferenceCheckOutput,
    model=secrets.gemini_api_model,
)

@output_guardrail
async def self_reference_output_guardrail(ctx: RunContextWrapper, agent: Agent, output: MessageOutput) -> GuardrailFunctionOutput:
    result = await Runner.run(self_reference_agent, output, context=ctx.context)
    data = result.final_output
    return GuardrailFunctionOutput(output_info=data, tripwire_triggered=data.contains_self_reference and not data.is_developer_context)
