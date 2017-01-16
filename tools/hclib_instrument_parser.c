#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hclib-instrument.h"

/*
 * A dump file is formatted as follows:
 *
 *  1. First line lists the number of event types contained in the dump file.
 *  2. The following N lines (where N is the number of event types) each list
 *     metadata on an event type, formatted as follows;
 *
 *         <event-type-id> <event-name>
 *
 *  3. After that, the remainder of the file is binary event data formatted as
 *     an array of hclib_instrument_event structs (which are defined in
 *     hclib-instrument.h).
 */
int main(int argc, char **argv) {
    int i;

    if (argc != 2) {
        fprintf(stderr, "usage: %s dump-file-dir\n", argv[0]);
        return 1;
    }

    DIR *dir = opendir(argv[1]);
    if (dir == NULL) {
        fprintf(stderr, "Failed opening directory %s\n", argv[1]);
        return 1;
    }

    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        char absolute_path[256];
        sprintf(absolute_path, "%s/%s", argv[1], ent->d_name);

        struct stat path_stat;
        stat(absolute_path, &path_stat);
        int is_file = S_ISREG(path_stat.st_mode);
        if (!is_file) continue;

        const int thread_id = atoi(ent->d_name);

        FILE *fp = fopen(absolute_path, "r");
        if (fp == NULL) {
            fprintf(stderr, "Failed opening %s\n", absolute_path);
            return 1;
        }

        char *line = NULL;
        size_t line_length = 0;
        size_t bytes_read = 0;

        ssize_t err = getline(&line, &line_length, fp);
        assert(err != -1);
        bytes_read += err;

        const unsigned n_event_types = atoi(line);
        hclib_event_type_info *event_info = (hclib_event_type_info *)malloc(
                n_event_types * sizeof(hclib_event_type_info));
        assert(event_info);

        for (i = 0; i < n_event_types; i++) {
            err = getline(&line, &line_length, fp);
            assert(err != -1);
            bytes_read += err;

            int index = 0;
            while (line[index] != ' ') index++;
            line[index] = '\0';

            const unsigned event_type_id = atoi(line);
            const char *event_name = line + index + 1;

            // Trim off newline character
            while (line[index] != '\n') index++;
            line[index] = '\0';

            event_info[i].event_type = event_type_id;
            event_info[i].name = (char *)malloc(strlen(event_name) + 1);
            assert(event_info[i].name);
            memcpy(event_info[i].name, event_name, strlen(event_name) + 1);
        }

        err = fseek(fp, 0L, SEEK_END);
        assert(err == 0);

        const long int file_size = ftell(fp);
        assert(file_size >= 0);

        const size_t buf_size = file_size - bytes_read;
        if (buf_size % sizeof(hclib_instrument_event) != 0) {
            fprintf(stderr, "Expected buf size (%lu) to be evenly divisible by "
                    "(%d) but was not. bytes_read = %lu, "
                    "sizeof(hclib_instrument_event) = %d\n", buf_size,
                    sizeof(hclib_instrument_event), bytes_read,
                    sizeof(hclib_instrument_event));
            return 1;
        }
        const size_t nevents = buf_size / sizeof(hclib_instrument_event);
        hclib_instrument_event *events = (hclib_instrument_event *)malloc(buf_size);
        assert(events);

        err = fseek(fp, bytes_read, SEEK_SET);
        assert(err == 0);

        err = fread(events, sizeof(hclib_instrument_event), nevents, fp);
        assert(err == nevents);

        for (i = 0; i < nevents; i++) {
            hclib_instrument_event *event = events + i;

            hclib_event_type_info *event_type = NULL;
            int j;
            for (j = 0; j < n_event_types; j++) {
                if (event_info[j].event_type == event->event_type) {
                    assert(event_type == NULL);
                    event_type = event_info + j;
                }
            }
            assert(event_type);

            char *transition_str = NULL;
            switch (event->transition) {
                case (START):
                    transition_str = "START";
                    break;
                case (END):
                    transition_str = "END";
                    break;
                default:
                    fprintf(stderr, "Unsupported transition type %d\n",
                            event->transition);
                    exit(1);
            }

            printf("%llu %d %s %s %d\n", event->timestamp_ns, thread_id,
                    event_type->name, transition_str, event->event_id);
        }

        fclose(fp);
    }

    closedir(dir);

    return 0;
}
