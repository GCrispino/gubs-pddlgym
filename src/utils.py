#def find_state_by_coord(x, y):
#    return [s for s in S if get_values(s.literals, 'robot-at')[0][1].split(':')[0][1:-1] == f'{x}-{y}'][0]
#
#    
#def get_values(obs, name):
#    values = []
#    for lit in obs:
#        if lit.predicate.name == name:
#            values.append(lit.variables)
#    return values

def tireworld_text_render(obs):
    vehicle_location = None
    flattire = True
    spare_in_locs = []
    for lit in obs.literals:
        if lit.predicate.name == 'vehicle-at':
            vehicle_location = lit.variables[0]
        elif lit.predicate.name == 'not-flattire':
            flattire = False
        elif lit.predicate.name == 'spare-in':
            spare_in_locs.append(lit.variables[0])
    return f"""
        Vehicle at {vehicle_location}
        Spare tires at {spare_in_locs}
        {"Flat tire" if flattire else ""}
    """

def expblocks_text_render(obs):
    clear = []
    ontable = []
    on = []
    holding = None
    destroyed_blocks = []
    table_destroyed = None
    for lit in obs.literals:
        if lit.predicate.name == 'clear':
            clear.append(lit.variables[0])
        elif lit.predicate.name == 'on':
            on.append(lit.variables[:2])
        elif lit.predicate.name == 'on-table':
            ontable.append(lit.variables[0])
        elif lit.predicate.name == 'destroyed':
            destroyed_blocks.append(lit.variables[0])
        elif lit.predicate.name == 'holding':
            holding = lit.variables[0]
        elif lit.predicate.name == 'table-destroyed':
            table_destroyed = lit.variables[0]
    return f"""
        {"Table destroyed" if table_destroyed else ""}
        {f"Holding {holding}" if holding else "Hand empty"}
        Clear blocks at {clear}
        Blocks on table: {ontable}
        Destroyed blocks: {destroyed_blocks}
        {on}
    """

text_render_env_functions = {
    "PDDLEnvTireworld-v0": tireworld_text_render,
    "PDDLEnvExplodingblocks-v0": expblocks_text_render
}

def text_render(env, obs):
    if env.spec.id not in text_render_env_functions:
        return ""
    return text_render_env_functions[env.spec.id](obs)

