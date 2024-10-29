from flask import Flask, render_template, redirect, url_for, request
from werkzeug.serving import run_simple
import os
import codecs
import glob
import utils


app = Flask(__name__)


FUNCS_1_MAN = [
    'linear_forward', 'linear_grad_W', 'linear_grad_b', 'sigmoid_forward',
    'sigmoid_grad_input', 'nll_forward', 'nll_grad_input',
    'tree_gini_index', 'tree_split_data_left', 'tree_split_data_right', 'tree_to_terminal'
]

FUNCS_1_OPT = []

FUNCS_2_MAN = [
    'dense_forward', 'dense_grad_input', 'dense_grad_W', 'dense_grad_b',
    'relu_forward', 'relu_grad_input',
    'l2_regularizer',
    'conv_matrix', 'box_blur', 'flatten_forward',
    'flatten_grad_input', 'maxpool_forward'
]

FUNCS_2_OPT = []


def is_complete(dict_, assignment):
    if assignment == 1:
        keys = FUNCS_1_MAN
    else:
        keys = FUNCS_2_MAN

    for k in keys:
        if not dict_.get(k, False):
            return False
    return True


def fancy_progress(dict_, assignment, yes='&#x2705;', no='&#10060;'):
    if assignment == 1:
        keys = FUNCS_1_MAN + FUNCS_1_OPT
    else:
        keys = FUNCS_2_MAN + FUNCS_2_OPT

    res = {}
    for key in keys:
        try:
            if dict_[key]:
                res[key] = yes
            else:
                res[key] = no
        except KeyError:
            res[key] = '-'
    return res


@app.route('/_admin/progress/done')
def progress_sort_done():
    assignment = int(request.args.get('assignment', 1))
    progress = utils.get_users_progress()
    user_dicts = []

    for user in utils.get_users_list():
        dict_ = {
            'is_complete': is_complete(progress[user], assignment=assignment),
            'username': user,
        }

        dict_.update(fancy_progress(progress[user], assignment=assignment))
        dict_.update(utils.get_user_info(user))
        user_dicts.append(dict_)

    user_dicts = sorted(user_dicts, key=lambda x: x['is_complete'])
    return render_template('index_assignment_{}.html'.format(assignment), users=user_dicts)


if __name__ == '__main__':
    run_simple('0.0.0.0', port=9999, application=app, use_reloader=True,
               reloader_interval=60*30, use_debugger=True)
    # run_simple('0.0.0.0', port=9999, application=app, use_reloader=True, use_debugger=True)
