import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The plug-in's notification queue, driven through the plug-in's own loop.
 *
 * Two review findings live in that loop and nowhere else, so nothing written in
 * Python can reach them: a batch that finished while ImageJ was measuring the
 * previous one was dropped when the GUI exited right after, and a notification
 * was deleted as it was read, which let the GUI start the next batch while the
 * previous one was still being measured.
 *
 * Run against the built jar, not against freshly compiled source:
 *
 *   javac --release 8 -encoding UTF-8 -cp "lib\ij.jar;dist\AutoWorm_ROI.jar" \
 *       -d <tmp> tests\AutoWormBridgeQueueTest.java
 *   java -cp "<tmp>;lib\ij.jar;dist\AutoWorm_ROI.jar" AutoWormBridgeQueueTest
 *
 * The stand-in for the GUI is this class in another process, told to sleep. It
 * needs to be alive while a batch is being handled and gone afterwards, and a
 * real second process is what makes that true rather than assumed.
 */
public class AutoWormBridgeQueueTest {

    public static void main(String[] args) throws Exception {
        if (args.length == 2 && "--sleep".equals(args[0])) {
            Thread.sleep(Long.parseLong(args[1]));
            return;
        }
        survivesTheGuiExiting();
        staysInTheQueueWhileMeasured();
        anUndeletableNotificationIsHandledOnce();
        anUnreadableNotificationIsDropped();
        aNotificationWrittenAsTheGuiExitsIsStillMeasured();
        System.out.println("AUTOWORM_BRIDGE_QUEUE_OK");
    }

    /**
     * A notification the GUI writes as it exits is measured, even when it lands
     * between the directory scan and the end of the pass.
     *
     * The window is small -- the GUI writes the file and exits as a single step,
     * and the loop looks again 150 ms later -- but it is the normal end of a
     * session, and the batch it loses is the last one. The loop used to ask
     * whether the process was alive *after* scanning, so a pass that found nothing
     * and then saw a dead process ended the session with that notification still
     * on disk: ROIs written, no measurements.
     *
     * Raced for real, this is a one-in-a-thousand timing that no test can hold
     * still, so the interleaving is modelled instead: a Process that writes the
     * notification at the moment it is asked whether it is still alive. That is
     * the whole of what the loop asks of a Process, so the model is the case.
     */
    private static void aNotificationWrittenAsTheGuiExitsIsStillMeasured() throws Exception {
        final Path bridge = Files.createTempDirectory("autoworm-queue-");
        final AtomicInteger measured = new AtomicInteger();
        final List<String> seen = new ArrayList<>();
        final boolean[] wrote = {false};
        Process gui = new Process() {
            @Override
            public boolean isAlive() {
                if (!wrote[0]) {
                    wrote[0] = true;
                    try {
                        writeNotification(bridge, 0, "complete");
                    } catch (Exception error) {
                        throw new IllegalStateException(error);
                    }
                }
                return false;
            }

            @Override public OutputStream getOutputStream() { return new ByteArrayOutputStream(); }
            @Override public InputStream getInputStream() { return new ByteArrayInputStream(new byte[0]); }
            @Override public InputStream getErrorStream() { return new ByteArrayInputStream(new byte[0]); }
            @Override public int waitFor() { return 0; }
            @Override public int exitValue() { return 0; }
            @Override public void destroy() { }
        };

        int count = Auto_Worm_ROI.pumpNotifications(bridge, gui, new Auto_Worm_ROI.BatchHandler() {
            @Override
            public Auto_Worm_ROI.Handled handle(Path notification, Properties values) {
                seen.add(notification.getFileName().toString());
                measured.incrementAndGet();
                return Auto_Worm_ROI.Handled.MEASURED;
            }
        });

        check(measured.get() == 1, "batches measured: expected 1, got " + measured.get());
        check(seen.equals(Arrays.asList(name(0))),
                "handled: expected [" + name(0) + "], got " + seen);
        check(leftOver(bridge).isEmpty(), "queue not empty: " + leftOver(bridge));
    }

    /**
     * A batch that finishes while the previous one is being measured survives the
     * GUI exiting straight afterwards.
     *
     * This is the case the queue exists for. The loop used to stop the moment the
     * GUI process was gone, and the queue directory was then deleted unread, so
     * that batch kept its ROIs and never got measurements.
     */
    private static void survivesTheGuiExiting() throws Exception {
        final Path bridge = Files.createTempDirectory("autoworm-queue-");
        writeNotification(bridge, 0, "complete");
        final Process gui = guiLikeProcess(1200);
        final List<String> seen = new ArrayList<>();
        int measured = Auto_Worm_ROI.pumpNotifications(bridge, gui,
                new Auto_Worm_ROI.BatchHandler() {
                    @Override
                    public Auto_Worm_ROI.Handled handle(Path notification, Properties values) {
                        seen.add(notification.getFileName().toString());
                        if (seen.size() == 1) {
                            try {
                                // Closing the window is the user's last action, and
                                // it happens while this batch is being measured.
                                gui.waitFor();
                                writeNotification(bridge, 5, "complete");
                            } catch (Exception error) {
                                throw new IllegalStateException(error);
                            }
                        }
                        return Auto_Worm_ROI.Handled.MEASURED;
                    }
                });
        check(measured == 2, "batches measured: expected 2, got " + measured);
        check(seen.equals(Arrays.asList(name(0), name(5))),
                "handled, in order: expected [" + name(0) + ", " + name(5) + "], got " + seen);
        check(leftOver(bridge).isEmpty(), "queue not empty: " + leftOver(bridge));
    }

    /**
     * The notification of the batch being measured is still in the queue for as
     * long as it is being measured.
     *
     * That file is what the GUI reads to decide whether it may start the next
     * batch. Deleting it as it is read opens that gate minutes too early, and a
     * second batch then writes ROIs into a folder still being measured.
     */
    private static void staysInTheQueueWhileMeasured() throws Exception {
        final Path bridge = Files.createTempDirectory("autoworm-queue-");
        writeNotification(bridge, 0, "complete");
        final Process gui = guiLikeProcess(300);
        final List<Boolean> presentWhileMeasuring = new ArrayList<>();
        int measured = Auto_Worm_ROI.pumpNotifications(bridge, gui,
                new Auto_Worm_ROI.BatchHandler() {
                    @Override
                    public Auto_Worm_ROI.Handled handle(Path notification, Properties values) {
                        try {
                            // What a measurement does to the queue: nothing, for a
                            // while.
                            gui.waitFor();
                            presentWhileMeasuring.add(Files.exists(notification));
                        } catch (Exception error) {
                            throw new IllegalStateException(error);
                        }
                        return Auto_Worm_ROI.Handled.MEASURED;
                    }
                });
        check(measured == 1, "batches measured: expected 1, got " + measured);
        check(presentWhileMeasuring.equals(Arrays.asList(Boolean.TRUE)),
                "in the queue while being measured: expected [true], got " + presentWhileMeasuring);
        check(leftOver(bridge).isEmpty(), "queue not empty after measuring: " + leftOver(bridge));
    }

    /**
     * A notification that cannot be deleted is measured once and no more.
     *
     * The file stays in the queue, so the loop keeps seeing it. Measuring the
     * same batch again on every pass would be worse than any delay, and the loop
     * still has to end.
     */
    private static void anUndeletableNotificationIsHandledOnce() throws Exception {
        final Path bridge = Files.createTempDirectory("autoworm-queue-");
        final Path stuck = writeNotification(bridge, 0, "complete");
        boolean settable = makeUndeletable(stuck);
        final Process gui = guiLikeProcess(200);
        final List<String> seen = new ArrayList<>();
        try {
            int measured = Auto_Worm_ROI.pumpNotifications(bridge, gui,
                    new Auto_Worm_ROI.BatchHandler() {
                        @Override
                        public Auto_Worm_ROI.Handled handle(Path notification, Properties values) {
                            seen.add(notification.getFileName().toString());
                            return Auto_Worm_ROI.Handled.MEASURED;
                        }
                    });
            check(measured == 1, "batches measured: expected 1, got " + measured);
            check(seen.size() == 1, "handled " + seen.size() + " times, expected 1");
            // The case is only really set up where the attribute could be set; the
            // rest of the test is about the loop ending and handling once, which
            // holds either way.
            if (settable)
                check(Files.exists(stuck),
                        "deleting this file was expected to fail, so it should still be here");
            System.out.println("  notification that cannot be deleted: " +
                    (settable ? "covered" : "not available on this system"));
        } finally {
            if (settable) Files.setAttribute(stuck, "dos:readonly", Boolean.FALSE);
            Files.deleteIfExists(stuck);
        }
    }

    /**
     * Sets the read-only attribute, which is how a notification that is readable
     * but will not go away is made on Windows: DeleteFile fails on a read-only
     * file, and returns before the plug-in's own error path. An open handle was
     * tried first and does not work -- the JDK opens files for shared deletion.
     */
    private static boolean makeUndeletable(Path file) {
        try {
            Files.setAttribute(file, "dos:readonly", Boolean.TRUE);
            return true;
        } catch (Exception unsupported) {
            return false;
        }
    }

    /**
     * A notification that cannot be read is dropped rather than retried.
     *
     * Nothing can ever read it, and leaving it in the queue would bring the loop
     * back to it every 150 ms for as long as the GUI is open -- and keep the GUI
     * from starting another batch for just as long.
     */
    private static void anUnreadableNotificationIsDropped() throws Exception {
        Path bridge = Files.createTempDirectory("autoworm-queue-");
        // Not ASCII, and the reader decodes as ASCII strictly. The GUI writes
        // base64, so this cannot happen from the GUI -- only from a truncated or
        // overwritten file, which is exactly what has to be survivable.
        Files.write(bridge.resolve(name(0)),
                new byte[] {(byte) 0xFF, (byte) 0xFE, 'a', '=', 'b'});
        Process gui = guiLikeProcess(200);
        final List<String> seen = new ArrayList<>();
        int measured = Auto_Worm_ROI.pumpNotifications(bridge, gui,
                new Auto_Worm_ROI.BatchHandler() {
                    @Override
                    public Auto_Worm_ROI.Handled handle(Path notification, Properties values) {
                        seen.add(notification.getFileName().toString());
                        return Auto_Worm_ROI.Handled.MEASURED;
                    }
                });
        check(measured == 0, "batches measured: expected 0, got " + measured);
        check(seen.isEmpty(), "an unreadable notification reached the handler: " + seen);
        check(leftOver(bridge).isEmpty(), "an unreadable notification was left behind");
    }

    /** A process that stands in for the GUI: alive for a while, then gone. */
    private static Process guiLikeProcess(long millis) throws Exception {
        String java = Paths.get(System.getProperty("java.home"), "bin", "java").toString();
        return new ProcessBuilder(java, "-cp", System.getProperty("java.class.path"),
                "AutoWormBridgeQueueTest", "--sleep", Long.toString(millis))
                .redirectErrorStream(true).inheritIO().start();
    }

    /** One notification, named and written the way the GUI does it. */
    private static Path writeNotification(Path bridge, int sequence, String status) throws Exception {
        Path file = bridge.resolve(name(sequence));
        Files.write(file, ("status=" + status + "\n").getBytes(StandardCharsets.US_ASCII));
        return file;
    }

    private static String name(int sequence) {
        return String.format("%020d.properties", sequence);
    }

    private static List<String> leftOver(Path bridge) throws Exception {
        List<String> left = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(bridge, "*.properties")) {
            for (Path path : stream) left.add(path.getFileName().toString());
        }
        return left;
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
