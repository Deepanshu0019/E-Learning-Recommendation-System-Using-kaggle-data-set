from learn_dev import create_app, db
from run_learn_dev import ensure_columns, seed_from_csv


def main():
    app = create_app()
    with app.app_context():
        db.create_all()
        ensure_columns(app)
        seed_from_csv(app)
    print("DB ensured and seeded.")


if __name__ == "__main__":
    main()


