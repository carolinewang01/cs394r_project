import pprint

from tic_tac_toe_iqn import get_args, train_agent, train_selfplay, watch


def test_tic_tac_toe(args=get_args()):
    if args.watch:
        watch(args)
        return

    #result, agent = train_agent(args)
    result, agent = train_selfplay(args)
    assert result["best_reward"] >= args.win_rate

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, agent_learn=agent)


if __name__ == '__main__':
    test_tic_tac_toe(get_args())
