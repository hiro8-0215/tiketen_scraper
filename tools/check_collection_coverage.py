"""Surface partial collection after preserving the successful checkpoints."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if report['expected_events'] != report['fetched_events'] or report['issues']:
        for issue in report['issues']:
            print('::warning title=Incomplete event coverage::' + json.dumps(issue, ensure_ascii=True))
        # Archive workflow is triggered only by successful collection jobs.
        # Keep the data and its backup, but make partial coverage visible.
        message = ('Partial collection: '
                   f"{report['fetched_events']}/{report['expected_events']} events; "
                   f"{len(report['issues'])} issues. Saved successful observations.")
        print(message)
        import os
        if summary := os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(summary, 'a', encoding='utf-8') as stream:
                stream.write('## ' + message + '\n\n')
                for issue in report['issues']:
                    stream.write('- ' + json.dumps(issue, ensure_ascii=True) + '\n')
        return
    print(f"API coverage complete: {report['fetched_events']} events")


if __name__ == '__main__':
    main()
