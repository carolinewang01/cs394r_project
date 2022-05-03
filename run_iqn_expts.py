import pprint


def test_tic_tac_toe():
    from tic_tac_toe_iqn_self_play import get_args, train_agent, train_selfplay, watch
    args = get_args()

    if args.watch:
        watch(args)
        return

    #result, agent = train_agent(args)
    result, agent = train_selfplay(args)
    #assert result["best_reward"] >= args.win_rate
    print("Result dictionary from last iteration of self-play.")
    pprint.pprint(result)
    # watch(args, agent_learn=agent)

def test_leduc():
    from leduc_iqn_random import get_args, train_agent, watch

    args=get_args()

    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    pprint.pprint(result)

if __name__ == '__main__':
    # test_tic_tac_toe()
    test_leduc()