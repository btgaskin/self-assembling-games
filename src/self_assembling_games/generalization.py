"""Single expansion of earned reinforcement across declared observation rows."""
import math

from .core import Address, Feedback


def reinforce_contexts(runtime, decision, outcome, coordinates, kernel):
    origin = runtime.tables[decision.table].context
    for context, coordinate in enumerate(coordinates):
        if kernel['kind'] == 'binary':
            weight = 1.0 if context == origin else kernel['alpha']
        else:
            width = kernel['width']
            weight = float(context == origin) if width == 0 else math.exp(
                -0.5 * ((coordinate - coordinates[origin]) / width) ** 2)
        if weight:
            runtime.feedback(Feedback(
                Address(runtime.table_id(decision.module, context), decision.option),
                outcome.success, outcome.payoff * weight, (decision.occurrence,),
            ))
