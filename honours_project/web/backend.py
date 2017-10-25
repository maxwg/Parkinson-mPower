import random
from flask import Flask, request, jsonify

results = {}
from celery import Celery
from web import diagnose


def make_celery(app):
    print(app.import_name)
    celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(app)
#
# @app.route('/<id>')
# def hello_world(id):
#     global results
#     print(results)
#     if id in results:
#         return results[id]
#     else:
#         return "result not available :("

@celery.task(name="tasks.diagnosis")
def perform_diagnosis(id, fname):
    print("DIAGNOSING",id,fname)
    print(results)
    from web.diagnose import diagnose
    res = diagnose(fname)
    return str(res)

@app.route('/online')
def online():
    return "yes"

@app.route('/<task_id>')
def taskstatus(task_id):
    task = perform_diagnosis.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'position': 0,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        return "There is a " + task.info + " probability of you having PD. This result is likely very inaccurate."
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

@app.route('/diagnose/', methods=['POST'])
def diagnose():
    id = str(random.randint(10000000000000, 9999999999999999))
    global results
    results[id] = "Your data is being processed!"
    filename = "tmp/" + id + ".wav"
    request.files['file'].save(filename)
    perform_diagnosis.apply_async(args=(id, filename), task_id=id)
    return id

if __name__ == '__main__':
    app.run()