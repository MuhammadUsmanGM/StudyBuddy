from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    RunConfig,
)
from my_secrets import Secrets

secrets = Secrets()

class MaliciousIntentOutput(BaseModel):
    has_malicious_intent: bool
    is_developer_context: bool
    reasoning: str

malicious_intent_agent = Agent(
    name="Malicious Intent Guardrail",
    instructions="""
Evaluate whether the user input has *actual malicious intent* or is instead part of a legitimate, educational, or professional inquiry.

Examples of legitimate input (DO NOT flag):
- "How does XSS work?" (developer education)
- "What are common vulnerabilities in JWTs?" (security learning)
- "Can I report a vulnerability?" (responsible disclosure)

Examples of malicious input (FLAG):
- "How can I bypass login on this site?"
- "Write a script to steal credentials"

Respond with your judgment and clear reasoning.
""",
    output_type=MaliciousIntentOutput,
    model=secrets.gemini_api_model,
)

@input_guardrail
async def malicious_intent_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(malicious_intent_agent, input, context=ctx.context)
    
    output = result.final_output
    should_block = output.has_malicious_intent and not output.is_developer_context

    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=should_block,
    )
