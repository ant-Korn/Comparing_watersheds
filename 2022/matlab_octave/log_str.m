function log_str(str, fid)
    fprintf(fid, "%s: %s\r\n", datestr(now, 0), str);
end